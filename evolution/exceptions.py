class NoneFitnessException(Exception):
    """Raised when an individual with a None fitness value is encountered."""
    def __init__(self, msg):
        self.message = msg
        super().__init__(self.message)