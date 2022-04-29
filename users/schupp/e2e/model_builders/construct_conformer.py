from e2e_modified_conformer import ConformerEncoder
import sys
import json

# Maybe this is risky but lets do it:
sys.setrecursionlimit(8000) # Previous was 1000

def conformer():
    conformer = ConformerEncoder()
    conformer.create_network()


    network = conformer.network.get_net()

    print(json.dumps(network,
        sort_keys=True, indent=4, default=str))

def conformer_conv_down():
    conformer = ConformerEncoder(
        input_layer="convp"
    )
    conformer.create_network()

    # With pooling on input 2x ( 2, 2) -> i.e.: time down = 4

    network = conformer.network.get_net()

    print(json.dumps(network,
        sort_keys=True, indent=4, default=str))



# 1 
# Just conformer, no downsampling
# conformer()


# 1 
# Just conformer, no vgg but pooling in subsampling

conformer_conv_down()