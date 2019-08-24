def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'sscnn':
        from .sscnn import SSCNN
        return SSCNN
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
