class TimeSignatureManager:
    def __init__(self, one_four_token=None, two_four_token=None, three_four_token=None, four_four_token=None,
                 five_four_token=None, six_four_token=None, seven_four_token=None, eight_four_token=None,
                 three_eight_token=None, five_eight_token=None, six_eight_token=None, seven_eight_token=None,
                 nine_eight_token=None, twelve_eight_token=None, two_two_token=None):
        self.two_two_token = two_two_token

        self.one_four_token = one_four_token
        self.two_four_token = two_four_token
        self.three_four_token = three_four_token
        self.four_four_token = four_four_token
        self.five_four_token = five_four_token
        self.six_four_token = six_four_token
        self.seven_four_token = seven_four_token
        self.eight_four_token = eight_four_token

        self.three_eight_token = three_eight_token
        self.five_eight_token = five_eight_token
        self.six_eight_token = six_eight_token
        self.seven_eight_token = seven_eight_token
        self.nine_eight_token = nine_eight_token
        self.twelve_eight_token = twelve_eight_token

    def is_valid_time_signature(self, n, d):
        return (d == 2 and n in [2]) or \
               (d == 4 and n in [1, 2, 3, 4, 5, 6, 7, 8]) or \
               (d == 8 and n in [3, 5, 6, 7, 9, 12])

    def from_fraction_to_time_and_token(self, n, d):
        if d == 2:
            if n == 2:
                return 2, self.two_two_token
            else:
                return None
        if d == 4:  # x/4 times
            if n == 1:
                return 1, self.one_four_token
            elif n == 2:
                return 2, self.two_four_token
            elif n == 3:
                return 3, self.three_four_token
            elif n == 4:
                return 4, self.four_four_token
            elif n == 5:
                return 5, self.five_four_token
            elif n == 6:
                return 6, self.six_four_token
            elif n == 7:
                return 7, self.seven_four_token
            elif n == 8:
                return 8, self.eight_four_token
            else:
                return None
        elif d == 8:  # x/8 times
            if n == 3:  # 3/8
                return 1.5, self.three_eight_token
            elif n == 5:  # 5/8
                return 2.5, self.five_eight_token
            elif n == 6:  # 6/8
                return 3, self.six_eight_token
            elif n == 7:
                return 3.5, self.seven_eight_token
            elif n == 9:  # 9/8
                return 4.5, self.nine_eight_token
            elif n == 12:  # 12/8
                return 6, self.twelve_eight_token
            else:
                return None
        else:
            return None

    def from_token_to_time_and_fraction(self, tok):
        if tok == self.one_four_token:
            return 1, 1, 4
        elif tok == self.two_four_token:
            return 2, 2, 4
        elif tok == self.three_four_token:
            return 3, 3, 4
        elif tok == self.four_four_token:
            return 4, 4, 4
        elif tok == self.five_four_token:
            return 5, 5, 4
        elif tok == self.six_four_token:
            return 6, 6, 4
        elif tok == self.seven_four_token:
            return 7, 7, 4
        elif tok == self.eight_four_token:
            return 8, 8, 4
        elif tok == self.three_eight_token:
            return 1.5, 3, 8
        elif tok == self.six_four_token:
            return 3, 6, 4
        elif tok == self.nine_eight_token:
            return 4.5, 9, 8
        elif tok == self.twelve_eight_token:
            return 6, 12, 8
        else:
            return None
