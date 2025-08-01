from src.classical_laminate_theory.failure_analysis.plot_failure_envelope import plot_fpf_failure_envelope
from src.classical_laminate_theory.failure_analysis.failure_envelope import FailureEnvelopeGenerator
from src.classical_laminate_theory.config_utils.config import CLTConfig
from src.classical_laminate_theory.config_utils.yaml_loader import load_config


def main(config: dict):

    clt_config = CLTConfig(config)

    labels = [criterion.name for criterion in clt_config.settings.failure_criteria]


    for layers in clt_config.layers_builder.build():
        print(f"Laminate: {[lay.rotation for lay in layers]}")
        # Create and plot envelope
        for load in clt_config.loading.loads:
            envelopes = [
                FailureEnvelopeGenerator(
                    analyser, load.x_axis, load.y_axis, angle_resolution=load.angle_resolution
                ).compute_envelope(layers)
                for analyser in clt_config.settings.analysers
            ]
            fig = plot_fpf_failure_envelope(envelopes, labels, x_label=load.x_axis, y_label=load.y_axis)

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

    for yaml_file in yaml_files:
        print(f"Running with: {yaml_file}")
        config = load_config(yaml_file)
        main(config)


# End