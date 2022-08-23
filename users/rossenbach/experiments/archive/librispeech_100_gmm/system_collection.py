from i6_experiments.common.setups.rasr.gmm_system import GmmSystem

gmm_systems_and_steps = {}


def add_gmm_system(key, system, steps):
    """
    :param str key:
    :param GmmSystem model:
    :param RasrSteps steps:
    """
    global gmm_systems_and_steps
    gmm_systems_and_steps[key] = (system, steps)


def get_system_and_steps(key):
    """

    :param key:
    :return:
    :rtype: (GmmSystem, RasrSteps)
    """
    return gmm_systems_and_steps[key]