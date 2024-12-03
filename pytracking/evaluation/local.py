from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.workspace_path = '/home/ubuntu/finalSubmissionRepo/SLTRLProject'  # Base directory for saving network checkpoints.
    settings.network_path = settings.workspace_path + '/checkpoints/ltr'
    settings.result_plot_path = settings.workspace_path+'/results/plot'
    settings.results_path = settings.workspace_path+'/results/tracking_results'
    settings.got_packed_results_path = settings.workspace_path+'/results/GOT-10k'
    settings.got10k_path = '/home/ubuntu/data/GOT-10K'

    return settings

