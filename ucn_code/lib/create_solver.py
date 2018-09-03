from caffe.proto import caffe_pb2

def getString(parameter, parameter_value, add_quotes=True):
    if type(parameter_value) == str:
        if add_quotes:
            return parameter + ': ' + '\"' + parameter_value + '\"' + '\n\n'
        else:
            return parameter + ': ' + parameter_value + '\n\n'
    else:
        return parameter + ': ' + str(parameter_value) + '\n\n'


def create(solver_params, prototxt):
    net = caffe_pb2.SolverParameter()
    solver_file = ''
    for keys in solver_params.keys():
        if (keys == 'solver_type') or (keys == 'solver_mode'):
            string = getString(keys, solver_params[keys], add_quotes=False)
        else:
            string = getString(keys, solver_params[keys])
        solver_file += string

    file_solver = open(prototxt, "w")
    file_solver.write(solver_file)
    file_solver.close()