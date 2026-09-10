import inspect
from typing import Dict, Any, Type, Callable, Optional, List

class Registry:
    """
    工业级通用插件注册中心。
    支持显式注册、装饰器注册、动态类型检查与基于 YAML 配置字典的实例化构造。
    """
    def __init__(self, name: str) -> None:
        if not isinstance(name, str) or len(name.strip()) == 0:
            raise ValueError("注册中心名称必须为非空的有效字符串！")
        self._name: str = name.strip()
        self._module_dict: Dict[str, Type] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def module_dict(self) -> Dict[str, Type]:
        return self._module_dict.copy()

    def _register_module(self, module_class: Type, module_name: Optional[str] = None, force: bool = False) -> None:
        if not inspect.isclass(module_class):
            raise TypeError(f"注册对象必须是一个类 (Class)，但接收到的类型为: {type(module_class)}")
        
        target_name = module_name if module_name is not None else module_class.__name__
        if not isinstance(target_name, str) or len(target_name.strip()) == 0:
            raise ValueError(f"模块名称必须为非空字符串，当前为: {target_name}")
        
        target_name = target_name.strip()
        if not force and target_name in self._module_dict:
            existing_class = self._module_dict[target_name]
            raise KeyError(
                f"模块名称 '{target_name}' 已经在注册表 '{self._name}' 中注册！"
                f"已有类: {existing_class.__module__}.{existing_class.__name__}, "
                f"尝试注册类: {module_class.__module__}.{module_class.__name__}"
            )
        self._module_dict[target_name] = module_class

    def register(self, name: Optional[str] = None, force: bool = False) -> Callable[[Type], Type]:
        """作为类装饰器使用"""
        def _register(cls: Type) -> Type:
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls
        return _register

    def get(self, name: str) -> Type:
        """根据模块名称获取对应的类对象"""
        if not isinstance(name, str):
            raise TypeError(f"查找的模块名称必须为字符串，当前接收到: {type(name)}")
        name = name.strip()
        if name not in self._module_dict:
            available = ", ".join(list(self._module_dict.keys())) if self._module_dict else "None"
            raise KeyError(
                f"未能在注册中心 '{self._name}' 中找到模块 '{name}'！\n"
                f"当前可用模块列表: [{available}]"
            )
        return self._module_dict[name]

    def build(self, cfg: Dict[str, Any], **default_args: Any) -> Any:
        """
        根据配置字典动态实例化对象。
        cfg 必须包含 'type' 字段以指定模块名。
        """
        if not isinstance(cfg, dict):
            raise TypeError(f"构建配置必须为字典 (dict)，当前接收到: {type(cfg)}")
        if "type" not in cfg:
            raise KeyError(f"构建配置字典中缺失必须的 'type' 键！当前配置内容: {cfg}")
        
        cfg_copy = cfg.copy()
        module_type = cfg_copy.pop("type")
        
        if not isinstance(module_type, str):
            raise TypeError(f"配置字典中的 'type' 必须为字符串类型，当前为: {type(module_type)}")
        
        cls_obj = self.get(module_type)
        
        combined_args = default_args.copy()
        combined_args.update(cfg_copy)
        
        try:
            instance = cls_obj(**combined_args)
            return instance
        except Exception as e:
            raise RuntimeError(
                f"在注册中心 '{self._name}' 中实例化模块 '{module_type}' 失败！\n"
                f"传入参数: {combined_args}\n"
                f"底层错误信息: {str(e)}"
            ) from e

    def list_modules(self) -> List[str]:
        return sorted(list(self._module_dict.keys()))

    def __contains__(self, name: str) -> bool:
        return name in self._module_dict

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, registered_modules={list(self._module_dict.keys())})"

# 全局核心业务注册中心单例
BACKBONE_REGISTRY = Registry("Backbone")       # 主干等变图网络
ATTENTION_REGISTRY = Registry("AttentionBias") # 物理引导注意力偏置
POTENTIAL_REGISTRY = Registry("Potential")     # 物理与量子势能
SCHEDULER_REGISTRY = Registry("Scheduler")     # 权重退火调度器
DATASET_REGISTRY = Registry("Dataset")         # 数据集加载器
