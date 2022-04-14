class DimensionError(IndexError):
  def __init__(self, expected, given, *args: object) -> None:
    self.message = f"Invalid dimensions. Expected {expected}, given {given}"
    super().__init__(self.message, *args)