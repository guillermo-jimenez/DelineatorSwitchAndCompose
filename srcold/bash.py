from pandas import read_csv
from argparse import ArgumentParser

if __name__ == '__main__':
    # Retrieve input data
    parser      = ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./Configurations.csv' , help='file with all considered configurations')
    parser.add_argument('--retrieve',    type=str, default=None, help='characteristic to retrieve, in text')
    parser.add_argument('--exec_ID',     type=int, help='select configuration ID')
    inputs      = parser.parse_args()
    config_path = inputs.config_path
    retrieve    = inputs.retrieve
    ID          = inputs.exec_ID

    # Read_CSV
    Configurations = read_csv(config_path,index_col=0)

    # Return value
    if retrieve in ["T_Dataset", "T_Dataset2", "A_Module", "T_Loss", "T_Optimizer", "A_KernelInitializer"]:
        print(Configurations.T[ID][retrieve]) # Stored in bash
    elif retrieve in ["A_LvlGrowth", "D_fs", "T_ValidationSplit", "T_LearningRate"]:
        print(float(Configurations.T[ID][retrieve])) # Stored in bash
    elif retrieve in ["T_DataAug", "A_MSUpsampling", "A_ASPP", "A_HyperDense", "A_MaxPooling"]:
        print(bool(Configurations.T[ID][retrieve])) # Stored in bash
    elif retrieve in ["T_Stride", "T_Epochs", "T_Stride2", "T_Epochs2", "A_Depth", "A_Repetitions", 
                      "A_InitChannels", "T_BatchSize", "A_KernelSize", "A_OutChannels", "P_ElementSize", 
                      "D_MaxSize", "T_Window", "T_Seed", "T_LRPatience", "T_Patience"]:
        print(int(Configurations.T[ID][retrieve])) # Stored in bash
    else:
        raise ValueError("Identifier does not exist") # Stored in bash


