"""Guard the standalone dynamics model against lost joints and false support."""
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import yaml

from jc04_dancing.resources import resource_path


@pytest.fixture(scope='module')
def model():
    path = resource_path('assets/robot/robot.xml')
    return mujoco.MjModel.from_xml_path(str(path))


def test_joints_and_actuators(model):
    # A fixed-to-world import or positional actuator mapping breaks this contract.
    original = ET.parse(resource_path('assets/robot/jc04.urdf')).getroot()
    names = {j.get('name') for j in original.findall('joint') if j.get('type') != 'fixed'}
    actual = {model.joint(i).name for i in range(model.njnt)}
    assert actual == names | {'root', 'right_assist', 'left_assist'}
    assert model.nu == 12
    assert np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE).tolist() == [model.joint('root').id]
    config = yaml.safe_load(resource_path('configs/robot.yaml').read_text())
    for name, settings in config['actuators'].items():
        actuator = model.actuator(name + '_motor')
        assert actuator.trnid[0] == model.joint(name).id
        assert actuator.ctrllimited
        np.testing.assert_allclose(actuator.ctrlrange, [-settings['max_torque'], settings['max_torque']])
        np.testing.assert_allclose(actuator.gear, [1, 0, 0, 0, 0, 0])
    assert model.equality('base_support').active0 == 0
    assert model.equality('base_support').type == mujoco.mjtEq.mjEQ_WELD
    for side in ('right', 'left'):
        brake = model.equality(side + '_wheel_lock')
        assert brake.active0 == 1
        assert brake.type == mujoco.mjtEq.mjEQ_JOINT
        assert brake.obj1id == model.joint(side + '_wheel_joint').id
        assert brake.obj2id == -1
    # Auxiliary wheels have physical bearings but no added control channels.
    for side in ('right', 'left'):
        assist = model.joint(side + '_assist')
        assert assist.type == mujoco.mjtJoint.mjJNT_HINGE
        assert assist.id not in model.actuator_trnid[:, 0]
        motor = model.actuator(side + '_wheel_joint_motor')
        assert motor.trnid[0] == model.joint(side + '_wheel_joint').id
    assert model.camera('overview').id >= 0


def test_inertias_meshes(model):
    # Dropped fixed links, inertias inferred from hulls, or mm-scale meshes fail.
    assert model.nmesh == 27
    assert model.body_mass.sum() == pytest.approx(21.812528045310945, rel=1e-7)
    moving = model.body_dofnum > 0
    assert np.all(model.body_mass[moving] > 0)
    inertia = model.body_inertia[moving]
    assert np.isfinite(inertia).all() and np.all(inertia > 0)
    assert np.all(2 * inertia.max(axis=1) <= inertia.sum(axis=1) + 1e-10)
    assert np.max(np.abs(model.mesh_vert)) < 1.0
    assert np.max(np.abs(model.mesh_vert)) > 0.1
    for geom in range(model.ngeom):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            assert model.geom_contype[geom] == model.geom_conaffinity[geom] == 0


def test_support_geometry(model):
    # Both wheel sets must carry loads, rather than touching a body hull.
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('standing').id)
    mujoco.mj_forward(model, data)
    heights = []
    for side in ('right', 'left'):
        for kind in ('driving', 'assist'):
            geom = model.geom(f'{side}_{kind}_contact')
            assert geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER
            assert geom.contype > 0 and geom.conaffinity > 0
            assert geom.size[0] == pytest.approx(0.091 if kind == 'driving' else 0.063, abs=0.003)
            axis = data.geom_xmat[geom.id].reshape(3, 3)[:, 2]
            assert abs(axis[1]) > 0.999
            heights.append(data.geom_xpos[geom.id, 2] - geom.size[0])
    assert min(heights) >= -1e-5
    assert max(heights) < 0.001
    assert data.qpos[2] > 0.35
    floor = model.geom('floor').id
    assert model.geom_type[floor] == mujoco.mjtGeom.mjGEOM_PLANE
    assert model.opt.timestep == pytest.approx(0.0005)
    np.testing.assert_allclose(model.opt.gravity, [0, 0, -9.81])
    np.testing.assert_allclose(model.geom_friction[floor], [1.0, 0.005, 0.0001])
    # Near-contact feet and no other body penetrating the ground.
    assert all(c.dist >= -1e-4 for c in data.contact)


def test_urdf_coordinates_and_positive_joint_directions(model):
    # Compare original URDF forward kinematics, including joint origins/axes,
    # against the exported model at rest and a +0.02 rad probe on each joint.
    from scipy.spatial.transform import Rotation
    data = mujoco.MjData(model)
    original = ET.parse(resource_path('assets/robot/jc04.urdf')).getroot()
    joints = original.findall('joint')
    probes = [None] + [j.get('name') for j in joints if j.get('type') != 'fixed']
    for probe in probes:
        mujoco.mj_resetData(model, data)
        if probe:
            data.qpos[model.joint(probe).qposadr[0]] += 0.02
        mujoco.mj_forward(model, data)
        transforms = {'base_link': np.eye(4)}
        transforms['base_link'][2, 3] = data.qpos[2]
        pending = list(joints)
        while pending:
            ready = [j for j in pending if j.find('parent').get('link') in transforms]
            assert ready, 'URDF tree has a cycle or detached body'
            for joint in ready:
                transform = np.eye(4)
                origin = joint.find('origin')
                transform[:3, 3] = np.fromstring(origin.get('xyz'), sep=' ')
                transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy'), sep=' ')).as_matrix()
                if joint.get('type') != 'fixed':
                    q = data.qpos[model.joint(joint.get('name')).qposadr[0]]
                    axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                    transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(axis * q).as_matrix()
                child = joint.find('child').get('link')
                transforms[child] = transforms[joint.find('parent').get('link')] @ transform
                pending.remove(joint)
        for index in range(1, model.nbody):
            expected = transforms[model.body(index).name]
            np.testing.assert_allclose(data.xpos[index], expected[:3, 3], atol=1e-11)
            np.testing.assert_allclose(data.xmat[index].reshape(3, 3), expected[:3, :3], atol=1e-11)


def test_conversion_uses_packaged_assets_and_rebuilds_from_other_cwd(tmp_path):
    # The CLI must compile without reading original deployment paths or cwd assets.
    import shutil
    robot = tmp_path / 'robot'
    shutil.copytree(resource_path('assets/robot'), robot, ignore=shutil.ignore_patterns('robot.xml'))
    output = robot / 'robot.xml'
    script = Path(__file__).resolve().parents[1] / 'tools/convert_model.py'
    result = subprocess.run([sys.executable, str(script), '--urdf', str(robot / 'jc04.urdf'), '--output', str(output)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rebuilt = mujoco.MjModel.from_xml_path(str(output))
    assert rebuilt.nu == 12
    assert output.read_bytes() == resource_path('assets/robot/robot.xml').read_bytes()


@pytest.mark.parametrize('name', ['robot_base', 'right_assist_wheel', 'left_assist_wheel'])
def test_large_mesh_conversion_preserves_triangles(name):
    # Native MSH avoids the STL face cap without deleting or rescaling faces.
    import struct
    source = resource_path(f'assets/robot/meshes/{name}.STL').read_bytes()
    count = struct.unpack_from('<I', source, 80)[0]
    dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    triangles = np.frombuffer(source, dtype=dtype, count=count, offset=84)['vertices']
    converted = resource_path(f'assets/robot/meshes/{name}.msh').read_bytes()
    nvert, nnorm, ntex, nface = struct.unpack_from('<4i', converted)
    assert (nnorm, ntex, nface) == (0, 0, count)
    vertices = np.frombuffer(converted, dtype='<f4', count=nvert * 3, offset=16).reshape(-1, 3)
    faces = np.frombuffer(converted, dtype='<i4', count=nface * 3, offset=16 + nvert * 12).reshape(-1, 3)
    np.testing.assert_array_equal(vertices[faces], triangles)
