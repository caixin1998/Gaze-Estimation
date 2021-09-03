from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # for model
        parser.add_argument('--visual_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        parser.add_argument('--phase', type=str, default="train", help='train|test')


        self.isTrain = True
        return parser
