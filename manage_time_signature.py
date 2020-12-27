from config import config


class TimeSignatureManager:
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def is_valid_time_signature(n, d):
        return (d == 2 and n in [2]) or \
               (d == 4 and n in [1, 2, 3, 4, 5, 6, 7, 8]) or \
               (d == 8 and n in [3, 5, 6, 7, 9, 12])

    @staticmethod
    def from_fraction_to_time_and_token(n, d):
        if d == 2:
            if n == 2:
                return 2, config["tokens"]["two_two"]
            else:
                return None
        if d == 4:  # x/4 times
            if n == 1:
                return 1, config["tokens"]["one_four"]
            elif n == 2:
                return 2, config["tokens"]["two_four"]
            elif n == 3:
                return 3, config["tokens"]["three_four"]
            elif n == 4:
                return 4, config["tokens"]["four_four"]
            elif n == 5:
                return 5, config["tokens"]["five_four"]
            elif n == 6:
                return 6, config["tokens"]["six_four"]
            elif n == 7:
                return 7, config["tokens"]["seven_four"]
            elif n == 8:
                return 8, config["tokens"]["eight_four"]
            else:
                return None
        elif d == 8:  # x/8 times
            if n == 3:  # 3/8
                return 1.5, config["tokens"]["three_eight"]
            elif n == 5:  # 5/8
                return 2.5, config["tokens"]["five_eight"]
            elif n == 6:  # 6/8
                return 3, config["tokens"]["six_eight"]
            elif n == 7:
                return 3.5, config["tokens"]["seven_eight"]
            elif n == 9:  # 9/8
                return 4.5, config["tokens"]["nine_eight"]
            elif n == 12:  # 12/8
                return 6, config["tokens"]["twelve_eight"]
            else:
                return None
        else:
            return None

    @staticmethod
    def from_token_to_time_and_fraction(tok):
        if tok == config["tokens"]["two_two"]:  # x/2
            return 2, 2, 2
        if tok == config["tokens"]["one_four"]:  # x/4
            return 1, 1, 4
        elif tok == config["tokens"]["two_four"]:
            return 2, 2, 4
        elif tok == config["tokens"]["three_four"]:
            return 3, 3, 4
        elif tok == config["tokens"]["four_four"]:
            return 4, 4, 4
        elif tok == config["tokens"]["five_four"]:
            return 5, 5, 4
        elif tok == config["tokens"]["six_four"]:
            return 6, 6, 4
        elif tok == config["tokens"]["seven_four"]:
            return 7, 7, 4
        elif tok == config["tokens"]["eight_four"]:
            return 8, 8, 4
        elif tok == config["tokens"]["three_eight"]:  # x/8
            return 1.5, 3, 8
        elif tok == config["tokens"]["five_eight"]:
            return 2.5, 5, 8
        elif tok == config["tokens"]["six_eight"]:
            return 3, 6, 4
        elif tok == config["tokens"]["seven_eight"]:
            return 3.5, 7, 8
        elif tok == config["tokens"]["nine_eight"]:
            return 4.5, 9, 8
        elif tok == config["tokens"]["twelve_eight"]:
            return 6, 12, 8
        else:
            return None
