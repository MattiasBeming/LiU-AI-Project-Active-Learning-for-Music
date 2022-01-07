# Run Code
When the environment has been set up, and all features and labels are available - it is possible to run the following processes:

- `model_selection` - performs active learning for all Learning Profiles, but the user input is simulated by using pre-defined annotations.
- `model_evaluation` - performs active learning on the best Learning Profiles from the `model_selection` process, this time with real user input.
- `presentation` - used for presenting results from either of the model processes in plots.

## How to Run
To run one of the processes, you must run `src/launch.py`. The first argument is always one of the processes mentioned above. For information on what parameters are required/available for each process, use the `--help` flag.

Example: \
`python src/launch.py model_selection --help`