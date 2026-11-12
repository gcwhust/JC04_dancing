"""Build JC04 MJCF using only the extracted URDF, meshes and robot.yaml.

Large binary STL assets are translated losslessly to native MSH. MuJoCo imports
the URDF and exports the inertias and fixed-body aggregation; this script adds
the explicit collision, scene, initial-pose and torque-control policy.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml


def numbers(values):
    return ' '.join(format(float(value), '.16g') for value in values)


def large_stl_to_msh(path):
    """Keep every source triangle and coordinate, only index shared vertices."""
    raw = path.read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0]
    if len(raw) != 84 + 50 * count:
        raise ValueError(f'Expected binary STL: {path.name}')
    if count <= 200000:
        return path
    dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    triangles = np.frombuffer(raw, dtype=dtype, count=count, offset=84)['vertices']
    vertices, indices = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    target = path.with_suffix('.msh')
    target.write_bytes(struct.pack('<4i', len(vertices), 0, 0, count)
                       + vertices.astype('<f4').tobytes()
                       + indices.astype('<i4').tobytes())
    return target


def convert(urdf, output):
    urdf = urdf.resolve()
    output = output.resolve()
    settings_path = Path(__file__).resolve().parents[1] / 'src/jc04_dancing/configs/robot.yaml'
    settings = yaml.safe_load(settings_path.read_text())
    source = ET.parse(urdf).getroot()
    # A massless base_link normally disappears into world during URDF fusion.
    # Make its original fixed child attachment free before MuJoCo imports it.
    attachment = source.find("joint[@name='base_joint']")
    if attachment is None or attachment.find('child').get('link') != 'robot_base':
        raise ValueError('Expected packaged JC04 base_joint -> robot_base')
    attachment.set('name', 'root')
    attachment.set('type', 'floating')
    for mesh in source.iter('mesh'):
        relative = Path(mesh.get('filename'))
        path = (urdf.parent / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(urdf.parent):
            raise ValueError(f'Mesh outside extracted asset directory: {relative}')
        converted = large_stl_to_msh(path)
        mesh.set('filename', converted.relative_to(urdf.parent).as_posix())
    extension = ET.SubElement(source, 'mujoco')
    ET.SubElement(extension, 'compiler', meshdir=str(urdf.parent), strippath='false',
                  discardvisual='false', fusestatic='true')
    imported = mujoco.MjModel.from_xml_string(ET.tostring(source, encoding='unicode'))
    with tempfile.TemporaryDirectory() as temporary:
        exported = Path(temporary) / 'import.xml'
        mujoco.mj_saveLastXML(str(exported), imported)
        scene = ET.parse(exported).getroot()
    scene.set('model', 'JC04 V1.3 standalone')
    compiler = scene.find('compiler')
    compiler.set('inertiafromgeom', 'false')
    option = ET.SubElement(scene, 'option', timestep=str(settings['scene']['timestep']),
                           gravity=numbers(settings['scene']['gravity']), integrator='implicitfast')
    option.set('cone', 'elliptic')
    # Procedural display assets are world-fixed and never enter collisions.
    asset = scene.find('asset')
    ET.SubElement(asset, 'texture', name='studio_sky', type='skybox', builtin='gradient',
                  rgb1='0.74 0.84 0.94', rgb2='0.94 0.97 1', width='512', height='3072')
    ET.SubElement(asset, 'texture', name='ground_checker', type='2d', builtin='checker',
                  rgb1='0.60 0.67 0.73', rgb2='0.82 0.87 0.91', width='512', height='512')
    ET.SubElement(asset, 'material', name='ground', texture='ground_checker',
                  texrepeat='2 2', texuniform='true', reflectance='0', shininess='0.05')
    ET.SubElement(scene, 'visual')
    ET.SubElement(scene.find('visual'), 'global', offwidth='1920', offheight='1080')
    ET.SubElement(scene.find('visual'), 'headlight', ambient='0.25 0.25 0.25', diffuse='0.55 0.55 0.55')
    world = scene.find('worldbody')
    # Keep imported visual meshes and source inertias, discard shell collisions.
    for body in world.iter('body'):
        # mj_saveLastXML prints six significant figures by default. Restore
        # full compiled precision for source kinematics and aggregate inertias.
        original_body = imported.body(body.get('name'))
        body.set('pos', numbers(original_body.pos))
        body.set('quat', numbers(original_body.quat))
        inertial = body.find('inertial')
        if inertial is not None:
            inertial.set('pos', numbers(original_body.ipos))
            inertial.set('quat', numbers(original_body.iquat))
            inertial.set('mass', numbers([original_body.mass[0]]))
            inertial.set('diaginertia', numbers(original_body.inertia))
        for index, geom in enumerate(list(body.findall('geom'))):
            if geom.get('type') != 'mesh' or geom.get('contype') != '0':
                body.remove(geom)
                continue
            geom.set('name', body.get('name') + '_visual_' + str(index))
            geom.set('group', '1')
            geom.set('contype', '0')
            geom.set('conaffinity', '0')
        for joint in body.findall('joint'):
            name = joint.get('name')
            original_joint = imported.joint(name)
            if original_joint.type[0] == mujoco.mjtJoint.mjJNT_HINGE:
                joint.set('pos', numbers(original_joint.pos))
                joint.set('axis', numbers(original_joint.axis))
                if original_joint.limited[0]:
                    joint.set('range', numbers(original_joint.range))
            if name in settings['actuators']:
                joint.set('damping', '0.05')
                # Bake the rest rotation into the body and compensate with ref.
                # qpos still denotes the original URDF angle, but qpos0 is rest.
                angle = math.radians(settings['rest_deg'][name])
                if angle:
                    axis = np.fromstring(joint.get('axis'), sep=' ')
                    delta = np.r_[math.cos(angle / 2), axis * math.sin(angle / 2)]
                    original = np.fromstring(body.get('quat', '1 0 0 0'), sep=' ')
                    product = np.empty(4)
                    mujoco.mju_mulQuat(product, original, delta)
                    body.set('quat', numbers(product))
                    joint.set('ref', str(angle))
    for side in ('right', 'left'):
        wheel = world.find(f".//body[@name='{side}_driving_wheel']")
        knee = world.find(f".//body[@name='{side}_knee_pitch_Link']")
        # Copy source primitive dimensions and the original fixed assist origin.
        for kind, body, position in [('driving', wheel, '0 0 0'), ('assist', knee,
                source.find(f"joint[@name='{side}_assist']/origin").get('xyz'))]:
            link = source.find(f"link[@name='{side}_{kind}_wheel']")
            cylinder = link.find('collision/geometry/cylinder')
            ET.SubElement(body, 'geom', name=f'{side}_{kind}_contact', type='cylinder',
                          pos=position, quat=numbers([math.sqrt(0.5), math.sqrt(0.5), 0, 0]),
                          size=f"{cylinder.get('radius')} {float(cylinder.get('length')) / 2}",
                          contype='1', conaffinity='1', group='3', rgba='0.1 0.5 0.9 0.3',
                          friction=numbers(settings['scene']['friction']), condim='4')
    # Small conservative primitives provide floor collision during falls. They
    # do not make a torso hull bridge the leg/wheel support or self-collide.
    base = world.find("body[@name='robot_base']")
    ET.SubElement(base, 'geom', name='torso_contact', type='box', pos='0 0 0.05',
                  size='0.09 0.10 0.17', contype='2', conaffinity='0', group='3', rgba='0.1 0.5 0.9 0.3')
    for side in ('right', 'left'):
        for part, endpoint in [('hip', '0 0 -0.21'), ('knee', '-0.02 0 -0.25')]:
            body = world.find(f".//body[@name='{side}_{part}_pitch_Link']")
            ET.SubElement(body, 'geom', name=f'{side}_{part}_contact', type='capsule',
                          fromto=f'0 0 -0.04 {endpoint}', size='0.025', contype='2',
                          conaffinity='0', group='3', rgba='0.1 0.5 0.9 0.3')
        elbow = world.find(f".//body[@name='{side}_elbow_pitch_Link']")
        ET.SubElement(elbow, 'geom', name=f'{side}_forearm_contact', type='capsule',
                      fromto='0 0 0 0.17 0 0', size='0.025', contype='2',
                      conaffinity='0', group='3', rgba='0.1 0.5 0.9 0.3')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='5 5 0.1',
                  rgba='1 1 1 1', material='ground', contype='1', conaffinity='3',
                  friction=numbers(settings['scene']['friction']))
    ET.SubElement(world, 'light', pos='1 -2 3', dir='-0.2 0.3 -1', directional='true', diffuse='0.65 0.65 0.65')
    ET.SubElement(world, 'camera', name='overview', pos='1.5 -2.1 1.3',
                  xyaxes='0.8137 0.5813 0 -0.163 0.228 0.960', fovy='38')
    motors = ET.SubElement(scene, 'actuator')
    for name, config in settings['actuators'].items():
        torque = config['max_torque']
        ET.SubElement(motors, 'motor', name=name + '_motor', joint=name, gear='1',
                      ctrllimited='true', ctrlrange=numbers([-torque, torque]))
    # Find base height directly from all four finite cylinder support functions.
    model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bottoms = []
    for side in ('right', 'left'):
        for kind in ('driving', 'assist'):
            geom = model.geom(f'{side}_{kind}_contact')
            axis_z = abs(data.geom_xmat[geom.id].reshape(3, 3)[2, 2])
            extent = geom.size[0] * math.sqrt(max(0, 1 - axis_z ** 2)) + geom.size[1] * axis_z
            bottoms.append(data.geom_xpos[geom.id, 2] - extent)
    height = -min(bottoms) + 0.0002
    base.set('pos', numbers([0, 0, height]))
    equality = ET.SubElement(scene, 'equality')
    ET.SubElement(equality, 'weld', name='base_support', body1='robot_base', active='false')
    wheels = settings['wheels']
    for name, joint in zip(wheels['lock_equalities'], wheels['joints']):
        ET.SubElement(equality, 'joint', name=name, joint1=joint,
                      polycoef='0 0 0 0 0', active=str(wheels['mode'] == 'locked').lower(),
                      solref='0.002 1', solimp='0.999 0.999 0.001')
    model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding='unicode'))
    keyframe = ET.SubElement(scene, 'keyframe')
    ET.SubElement(keyframe, 'key', name='standing', qpos=numbers(model.qpos0))
    compiler.set('meshdir', os.path.relpath(urdf.parent, output.parent))
    ET.indent(scene, space='  ')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(ET.tostring(scene, encoding='unicode') + '\n')
    # The source manifest keeps its original entries; append derived assets only
    # when working in the packaged assets tree. No original checkout is opened.
    manifest_path = urdf.parent.parent / 'provenance.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        derived = []
        for mesh in sorted(urdf.parent.glob('meshes/*.msh')):
            original = mesh.with_suffix('.STL')
            derived.append({'source': 'robot/meshes/' + original.name,
                            'target': 'robot/meshes/' + mesh.name,
                            'source_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                            'target_sha256': hashlib.sha256(mesh.read_bytes()).hexdigest(),
                            'transformation': 'Lossless binary STL to native MSH; retain all triangles and coordinates'})
        if output.parent == urdf.parent:
            derived.append({'source': 'robot/jc04.urdf', 'target': 'robot/robot.xml',
                            'source_sha256': hashlib.sha256(urdf.read_bytes()).hexdigest(),
                            'target_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                            'transformation': 'MuJoCo 3.3.7 URDF import and MJCF export; scene and contacts: docs/model-conversion.md'})
        targets = {entry['target'] for entry in derived}
        manifest['files'] = sorted([entry for entry in manifest['files'] if entry['target'] not in targets] + derived,
                                   key=lambda entry: entry['target'])
        manifest['model']['converted_with'] = 'MuJoCo ' + mujoco.__version__
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote {output.name}: {model.njnt} joints, {model.nu} motors; mass '
          f'{model.body_mass.sum():.12f} kg; standing base z={height:.9f} m')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    convert(args.urdf, args.output)


if __name__ == '__main__':
    main()
