class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


def read_training_data(file_path):
    training_data = []
    with open(file_path, 'r') as file:
        content = file.read()
        parts = content.split(';')

        i = 0
        while i < len(parts) - 4:
            try:
                # Read the index (skip it)
                entry_index = int(parts[i].strip())
                i += 1

                # Read the matrix vector
                matrix_vector_str = parts[i].strip()
                matrix_vector = [int(x) for x in matrix_vector_str.split(',')]
                if len(matrix_vector) != 81:
                    raise ValueError("Matrix vector length is not 81")
                matrix = [matrix_vector[j:j+9] for j in range(0, 81, 9)]
                i += 1

                # Read the vector
                vector_str = parts[i].strip()
                vector = [float(x) for x in vector_str.split(',')]
                i += 1

                # Read the two consecutive values
                value1 = int(parts[i].strip())
                value2 = int(parts[i + 1].strip())
                i += 2

                # Append the formatted data to the training_data list
                training_data.append((matrix, vector, value1, value2))
            except (ValueError, IndexError):
                # Skip malformed entries
                i += 1

    return training_data