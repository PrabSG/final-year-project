from typing import Iterable, Set, Tuple, Union


class SafetyRequirement():
  def __init__(self, formula: Union[Tuple, str], avoid_objs: Iterable[str] = [], req_objs: Iterable[str] = []) -> None:
    self._formula = formula
    self._avoid_objs = set(avoid_objs)
    self._req_objs = set(req_objs)

  def get_formula(self) -> Union[Tuple, str]:
    return self._formula

  def get_avoid_objs(self) -> Set[str]:
    return self._avoid_objs

  def get_req_objs(self) -> Set[str]:
    return self._req_objs

class AvoidRequirement(SafetyRequirement):
  def __init__(self, formula_to_avoid: Union[Tuple, str], avoid_objs: Iterable[str] = [], req_objs: Iterable[str] = []) -> None:
    super().__init__(formula_to_avoid, avoid_objs, req_objs)
  
  def get_formula(self) -> Union[Tuple, str]:
    return ("not", self._formula)

class UntilRequirement(SafetyRequirement):
  def __init__(self, req: SafetyRequirement, until_req: SafetyRequirement):
    self._req = req
    self._until_req = until_req

  def get_formula(self) -> Union[Tuple, str]:
    return ("until", self._req.get_formula(), self._until_req.get_formula())
  
  def get_avoid_objs(self) -> Set[str]:
    return self._req.get_avoid_objs().union(self._until_req.get_avoid_objs())
  
  def get_req_objs(self) -> Set[str]:
    return self._req.get_req_objs().union(self._until_req.get_req_objs())
