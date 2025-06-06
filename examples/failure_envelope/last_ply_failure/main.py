from classical_laminate_theory.clt.failure_analysis.failure_envelope import FailureEnvelopeGenerator
from classical_laminate_theory.clt.failure_analysis.plot_failure_envelope import plot_lpf_failure_envelope
from classical_laminate_theory.clt.config.config import CLTConfig
from classical_laminate_theory.clt.config.yaml_loader import load_config


def main(config: dict):

    config_class = CLTConfig(config)

    layers = config_class.create_layers()

    # Create and plot envelope
    for load in config_class.loading.loads:
        analyser = config_class.settings.analysers[0]
        envelope = FailureEnvelopeGenerator(analyser, load.x_axis, load.y_axis, angle_resolution=load.angle_resolution).compute_envelope(layers)
        fig = plot_lpf_failure_envelope(envelope, x_label=load.x_axis, y_label=load.y_axis)
        fig.show()
    

if __name__ == "__main__":
    import sys
    import os
    import glob

    if len(sys.argv) < 2:
        print("\nNo .yaml files provided.")
        print("Suggested usage: python main.py <path_to_yaml_config>")
        print("Now proceeding to executing all .yaml files...\n")
        
        # Get the directory of the script being executed
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Find all YAML files in the script's directory
        yaml_files = glob.glob(os.path.join(script_dir, "*.yaml")) + glob.glob(os.path.join(script_dir, "*.yml"))
    else:
        yaml_files = sys.argv[1:]

    yaml_file = yaml_files[0]
    config = load_config(yaml_file)
    main(config)


# End
 