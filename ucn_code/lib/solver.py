import math
import __init__paths
import caffe
import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format

from utils.timer import Timer


class Solver(object):

    def __init__(self, solver_prototxt, output_dir, pretrained_model=None, solver_state=None):

        self.output_dir = output_dir

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            text_format.Merge(f.read(), self.solver_param)

        if self.solver_param.solver_type == \
                caffe_pb2.SolverParameter.SolverType.Value('SGD'):
            self.solver = caffe.SGDSolver(solver_prototxt)
        elif self.solver_param.solver_type == \
                caffe_pb2.SolverParameter.SolverType.Value('NESTEROV'):
            self.solver = caffe.NesterovSolver(solver_prototxt)
        elif self.solver_param.solver_type == \
                caffe_pb2.SolverParameter.SolverType.Value('ADAGRAD'):
            self.solver = caffe.AdaGradSolver(solver_prototxt)
        elif self.solver_param.solver_type == \
                caffe_pb2.SolverParameter.SolverType.Value('ADAM'):
            self.solver = caffe.AdamSolver(solver_prototxt)
        else:
            raise NotImplementedError('Solver type not defined')

        if pretrained_model is not None:
            print('Loading pretrained model '
                  'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        if solver_state is not None:
            self.solver.restore(solver_state)


    def train(self, max_iters):
        """Network training loop."""

        timer = Timer()

        self.loss = np.zeros((max_iters,))
        iter_ = 0


        while self.solver.iter < max_iters:

            timer.tic()
            self.solver.step(1)
            timer.toc()

            self.loss[iter_] = self.solver.loss


            iter_ += 1

