from typing import List, Union, Tuple

import torch

UNARY_OPERATORS = ["not", "next"]
BINARY_OPERATORS = ["and", "or", "until"]
MISC_SYMBOLS = ["True", "False", "(", ")"]

OPERATOR_MAPPING = {
  "not": 0,
  "next": 1,
  "and": 2,
  "or": 3,
  "until": 4,
  "True": 5,
  "False": 6,
  "(": 7,
  ")": 8
}

def safety_spec_to_str(spec: Union[Tuple, str]) -> List[str]:
  # Base case
  if isinstance(spec, str):
    return [spec]
  
  # Recursive case
  spec_strs = []
  spec_strs.append("(")
  op = spec[0] # Operator always in first position in Tuple
  if op in UNARY_OPERATORS:
    spec_strs.append(op)
    spec_strs += safety_spec_to_str(spec[1])
  elif op in BINARY_OPERATORS:
    spec_strs += safety_spec_to_str(spec[1])
    spec_strs.append(op)
    spec_strs += safety_spec_to_str(spec[2])
  else:
    raise NotImplementedError(f"Unrecognised logical operator {op} in {spec}.")
  spec_strs.append(")")
  return spec_strs

def _prop_mapping(prop: str):
  return ord(prop) - ord("a")

def get_one_hot_spec(safety_spec: List[str], num_props: int) -> torch.Tensor:
  vocab_size = get_encoding_size(num_props)
  one_hot_encoding = torch.zeros((len(safety_spec), vocab_size), dtype=torch.float)

  for i, word in enumerate(safety_spec):
    if word in OPERATOR_MAPPING:
      one_hot_encoding[i][OPERATOR_MAPPING[word] + num_props] = 1
    else:
      prop_encoding = _prop_mapping(word)
      if prop_encoding >= num_props:
        raise ValueError(f"Too many propositions given, expected: {num_props}")
      one_hot_encoding[i][prop_encoding] = 1

  return one_hot_encoding

def get_encoding_size(num_props: int) -> int:
  return num_props + len(UNARY_OPERATORS) + len(BINARY_OPERATORS) + len(MISC_SYMBOLS)

if __name__ == "__main__":
  ex_spec = (
    "not", (
      "and",
      (
        "or", "b", "c"
      ),
      "a"
    )
  )
  spec_str = safety_spec_to_str(ex_spec)
  print(spec_str)
  print(get_one_hot_spec(spec_str, 3))
