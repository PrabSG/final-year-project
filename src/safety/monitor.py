from safety.dfa import DFA


class SafetyMonitor:
  def __init__(self, safety_spec):
    self._dfa = DFA(safety_spec)
    self._current_formula = safety_spec
    self._violation_count = 0
  
  def step(self, true_props):
    """Check the safety specification is violated by the true propositions."""
    progression = self._dfa.progress_LTL(self._current_formula, true_props)
    if progression == 'True':
      # All safety specs satisfied for some reason
      self._current_formula = progression
    elif progression == 'False':
      # Violated specification
      self._violation_count += 1
      return True
    else:
      self._current_formula = progression
    
    return False
  
  def get_safety_spec(self):
    return self._dfa.formula

  @property
  def violation_count(self):
    return self._violation_count

  