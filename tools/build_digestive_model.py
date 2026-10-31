"""Build a lossless, recolored teaching GLB from the supplied digestive export.

Usage: python3 tools/build_digestive_model.py SOURCE.glb OUTPUT.glb
Only scene-referenced geometry is retained. Vertex/index bytes and node transforms
are preserved; materials, grouping and provenance metadata are adapted.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

GROUPS = {
    'gallbladder': ('쓸개', 9024105),
    'large-intestine': ('큰창자', 12226179),
    'liver': ('간', 11429979),
    'oesophagus': ('식도', 14459800),
    'pancreas': ('이자', 14858348),
    'small-intestine': ('작은창자', 14263969),
    'stomach': ('위', 15638693),
}
LABELS = {
    'Gallbladder': '쓸개', 'Ascending colon': '오름잘록창자',
    'Descending colon': '내림잘록창자', 'Sigmoid colon': '구불잘록창자',
    'Transverse colon': '가로잘록창자', 'Vermiform appendix': '충수',
    'Liver': '간', 'Oesophagus': '식도', 'Pancreas': '이자',
    'Duodenum': '샘창자', 'Jejunum': '빈창자', 'Stomach': '위',
}
COMPONENT_BYTES = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
COMPONENTS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


def read_glb(path):
    raw = path.read_bytes()
    assert struct.unpack_from('<4sII', raw) == (b'glTF', 2, len(raw))
    chunks = {}
    offset = 12
    while offset < len(raw):
        length, kind = struct.unpack_from('<I4s', raw, offset)
        assert length % 4 == 0 and offset + 8 + length <= len(raw)
        chunks[kind] = raw[offset + 8:offset + 8 + length]
        offset += 8 + length
    return json.loads(chunks[b'JSON']), chunks[b'BIN\x00']


def accessor_bytes(doc, binary, index):
    accessor = doc['accessors'][index]
    assert 'sparse' not in accessor and not accessor.get('extensions')
    view = doc['bufferViews'][accessor['bufferView']]
    assert view['buffer'] == 0 and not view.get('extensions')
    size = COMPONENT_BYTES[accessor['componentType']] * COMPONENTS[accessor['type']]
    stride = view.get('byteStride', size)
    start = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
    end = start + (accessor['count'] - 1) * stride + size
    assert end <= view.get('byteOffset', 0) + view['byteLength']
    return b''.join(binary[start + i * stride:start + i * stride + size]
                    for i in range(accessor['count']))


def srgb_linear(value):
    value /= 255
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def build(source, target):
    doc, binary = read_glb(source)
    # This exporter intentionally supports static, uncompressed triangle models.
    # Reject unsupported content rather than silently dropping it.
    assert not any(doc.get(k) for k in ('extensionsUsed', 'skins', 'animations', 'textures', 'images'))
    scene = doc['scenes'][doc.get('scene', 0)]
    node_ids = scene['nodes']
    assert len(set(node_ids)) == len(node_ids)
    assert all(not doc['nodes'][i].get('children') for i in node_ids)
    result = {'asset': {'version': '2.0', 'generator': 'digest teaching model builder 1.0'},
              'scene': 0, 'scenes': [{'name': '소화계 학습 모델', 'nodes': []}],
              'nodes': [], 'meshes': [], 'materials': [], 'accessors': [],
              'bufferViews': [], 'buffers': []}
    result['asset']['extras'] = {
        'sourceFile': source.name,
        'sourceSHA256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'sourceAsset': deepcopy(doc['asset']),
        'sourceProjectsReportedByUser': ['Z-Anatomy', 'BodyParts3D'],
        'sourceLicenseMetadataPresent': 'copyright' in doc['asset'],
        'changes': ['Retained only scene-referenced geometry',
                    'Losslessly repacked vertex and index buffers',
                    'Applied organ colors and nonmetallic materials',
                    'Added identity-transform organ groups and Korean labels'],
        'geometryPolicy': 'Vertex/index bytes and anatomical node transforms unchanged; no organs synthesized',
    }
    out_binary = bytearray()
    accessor_map = {}
    group_nodes = {}
    group_materials = {}
    counts = Counter()
    for group, (label, color) in GROUPS.items():
        group_materials[group] = len(result['materials'])
        result['materials'].append({
            'name': label,
            'doubleSided': True,
            'pbrMetallicRoughness': {
                'baseColorFactor': [srgb_linear((color >> shift) & 255) for shift in (16, 8, 0)] + [1],
                'metallicFactor': 0, 'roughnessFactor': 0.7}})
        group_nodes[group] = len(result['nodes'])
        result['scenes'][0]['nodes'].append(group_nodes[group])
        result['nodes'].append({'name': label, 'children': [],
                                'extras': {'organ': label, 'anatomyId': group}})

    def copy_accessor(index):
        if index in accessor_map:
            return accessor_map[index]
        data = accessor_bytes(doc, binary, index)
        out_binary.extend(b'\x00' * (-len(out_binary) % 4))
        view_index = len(result['bufferViews'])
        original = doc['bufferViews'][doc['accessors'][index]['bufferView']]
        view = {'buffer': 0, 'byteOffset': len(out_binary), 'byteLength': len(data)}
        if 'target' in original:
            view['target'] = original['target']
        result['bufferViews'].append(view)
        out_binary.extend(data)
        accessor = deepcopy(doc['accessors'][index])
        accessor.update(bufferView=view_index, byteOffset=0)
        accessor_map[index] = len(result['accessors'])
        result['accessors'].append(accessor)
        return accessor_map[index]

    for node_id in node_ids:
        node = deepcopy(doc['nodes'][node_id])
        extras = node.get('extras', {})
        anatomy_id = extras['anatomyId']
        group = next(group for group in GROUPS
                     if anatomy_id == group or anatomy_id.startswith(group + '-'))
        label = GROUPS[group][0]
        node['extras'] = {k: v for k, v in extras.items()
                          if k in ('anatomyId', 'anatomyParentId', 'sourceName', 'anatomySystem')}
        node['extras'].update(organ=label, labelKo=LABELS[extras['sourceName']])
        source_mesh = doc['meshes'][node['mesh']]
        mesh = deepcopy(source_mesh)
        for primitive in mesh['primitives']:
            assert primitive.get('mode', 4) == 4
            assert not primitive.get('extensions') and not primitive.get('targets')
            primitive['attributes'] = {name: copy_accessor(i)
                                       for name, i in primitive['attributes'].items()}
            primitive['indices'] = copy_accessor(primitive['indices'])
            primitive['material'] = group_materials[group]
            counts['triangles'] += result['accessors'][primitive['indices']]['count'] // 3
        node['mesh'] = len(result['meshes'])
        result['meshes'].append(mesh)
        result['nodes'][group_nodes[group]]['children'].append(len(result['nodes']))
        result['nodes'].append(node)
        counts['parts'] += 1
    result['buffers'].append({'byteLength': len(out_binary)})
    json_chunk = json.dumps(result, ensure_ascii=False, separators=(',', ':')).encode()
    json_chunk += b' ' * (-len(json_chunk) % 4)
    out_binary.extend(b'\x00' * (-len(out_binary) % 4))
    output = (struct.pack('<4sII', b'glTF', 2, 28 + len(json_chunk) + len(out_binary))
              + struct.pack('<I4s', len(json_chunk), b'JSON') + json_chunk
              + struct.pack('<I4s', len(out_binary), b'BIN\x00') + out_binary)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(output)
    # Verify the serialized artifact, not just the in-memory conversion.
    checked, checked_binary = read_glb(target)
    for old, new in accessor_map.items():
        assert accessor_bytes(doc, binary, old) == accessor_bytes(checked, checked_binary, new)
    for source_id, converted in zip(node_ids, checked['nodes'][len(GROUPS):]):
        original = doc['nodes'][source_id]
        for key in ('translation', 'rotation', 'scale', 'matrix'):
            assert original.get(key) == converted.get(key)
    print(f'Verified {counts["parts"]} parts, {len(GROUPS)} groups, {counts["triangles"]:,} triangles; geometry bytes and transforms unchanged.')
    print(f'{source.stat().st_size:,} → {target.stat().st_size:,} bytes: {target}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error('Source and output must be different files.')
    build(args.source, args.output)
