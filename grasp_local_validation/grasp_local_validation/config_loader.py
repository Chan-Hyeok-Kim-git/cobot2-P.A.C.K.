"""
grasp_description 패키지의 share 디렉토리에서 rg2_collision.yaml / shelf_collision.yaml 을 읽어온다.
로컬(순수 python) 테스트를 위해 직접 경로도 받을 수 있게 fallback 제공.
"""
import os
import yaml

try:
    from ament_index_python.packages import get_package_share_directory
    _HAS_AMENT = True
except ImportError:
    _HAS_AMENT = False


def _load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_rg2_config(override_path=None):
    if override_path:
        return _load_yaml(override_path)
    if not _HAS_AMENT:
        raise RuntimeError("ament_index_python 없음. override_path로 직접 경로 지정 필요")
    share = get_package_share_directory('grasp_description')
    return _load_yaml(os.path.join(share, 'config', 'rg2_collision.yaml'))


def load_shelf_config(override_path=None):
    if override_path:
        return _load_yaml(override_path)
    if not _HAS_AMENT:
        raise RuntimeError("ament_index_python 없음. override_path로 직접 경로 지정 필요")
    share = get_package_share_directory('grasp_description')
    return _load_yaml(os.path.join(share, 'config', 'shelf_collision.yaml'))
