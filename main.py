from absl import app
from absl import flags
import ml_collections
from ml_collections.config_flags import config_flags
import wandb
import runner
import wandb

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)

flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum(
    "mode", None, ["train", "eval", "score", "sweep"], "Running mode: train or eval"
)
flags.DEFINE_string(
    "eval_folder", "eval", "The folder name for storing evaluation results"
)
flags.DEFINE_string(
    "sweep_id", None, "Optional ID for a sweep controller if running a sweep."
)
flags.DEFINE_string("project", None, "Wandb project name.")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
    config = FLAGS.config
    workdir = FLAGS.workdir

    if FLAGS.mode == "sweep":

        def train_sweep(sweep_id):

            with wandb.init():
                # Process config params to ml dict
                config = FLAGS.config.to_dict()
                sweep_config = wandb.config
                

                for p, val in sweep_config.items():
                    # First '_' splits into upper level
                    keys = p.split("_")
                    # print(keys)
                    parent = keys[0]
                    child = "_".join(keys[1:])
                    config[parent][child] = val

                wandb.config.update(config)
                config = ml_collections.ConfigDict(wandb.config)
                print(config)
                print(f"SWEEPID in train_sweep:{sweep_id}")
                runner.train(config, workdir, sweep_id=sweep_id)
            
            return
        
        config = FLAGS.config.to_dict()
        sweep_config = config["sweep"]
        
        if FLAGS.sweep_id is not None:
            sweep_id = FLAGS.sweep_id
            print(f"SWEEPID is not none:{sweep_id}")
        else:
            sweep_id = wandb.sweep(sweep_config, project="categorical")
            print(f"SWEEPID is made:{sweep_id}")

        # Start sweep job
        wandb.agent(sweep_id, lambda: train_sweep(sweep_id), project="categorical", count=20)

    elif FLAGS.mode == "eval":
        print(f"Config contents from main: {config}")
        runner.eval(config, workdir)
    else:
        wandb.init(project="categorical", config=config.to_dict(), resume="allow")
        print(config)
        runner.train(config, workdir)


if __name__ == "__main__":
    app.run(main)
