from neural import Neural_Network


def main() -> None:
    neural= Neural_Network(csv_flie_path='../treino_sinais_vitais_com_label.txt')
    neural.creating_model()

    return


main()
