import torch
class Config:
    drug_data_path = "Datasets/ECFP.csv"
    target_data_path = ("Datasets/Target.csv")
    cell_data_path = "Datasets/Gene.csv"
    interaction_data_path = "Datasets/samples/samples.csv"
    best_model_path = "best_model.pth"
    drug_graph_feature_path = "Datasets/Graph.csv"

    # 模型参数
    num_layers = 2
    node_dim = 256
    signnet_hidden = 512
    n_heads = 4
    ff_dim = 512
    max_degree = 500
    max_path_length = 500
    max_path_samples = 10
    lap_pe_dim = node_dim
    max_freqs = 10

    cell_dim=512

    lr = 0.0001
    epochs = 40
    batch_size = 256
    result_file = "results.csv"
    gated_gcn_dropout = 0.4
    gated_gcn_batchnorm = True


    device = "cuda" if torch.cuda.is_available() else "cpu"
