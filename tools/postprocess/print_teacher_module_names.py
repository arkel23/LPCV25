from fgir_kd.model_utils.build_model import build_model
from fgir_kd.other_utils.build_args import parse_train_args
from fgir_kd.data_utils.build_dataloaders import build_dataloaders


def main():
    args = parse_train_args()
    train_loader, val_loader, test_loader = build_dataloaders(args)

    model_t = build_model(args, teacher=True)
    for name, layer in model_t.named_modules():
        print(name)

    return 0


if __name__ == '__main__':
    main()

