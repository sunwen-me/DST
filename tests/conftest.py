"""pytest 公共配置: 注册自定义标记 (tests/ 目录本身由 conftest 机制加入 sys.path,
使 test_* 文件可直接 `import synthetic_scene`)。"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: 慢测 (完整论文形态: tiny 训练 + fast/both 模式端到端)")
