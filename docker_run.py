"""
Build locally:
docker build -t gcr.io/dexterous-manipulation-238516/dex-image:{VERSION}

Push to GCP:
docker push gcr.io/dexterous-manipulation-238516/dex-image:{VERSION}

Create GCP cluster (can't use GPU for free tier):
gcloud container clusters create [CLUSTER_NAME] \
--accelerator type=[GPU_TYPE],count=[AMOUNT] \
--region [REGION] --node-locations [ZONE],[ZONE]


"""
from common.constants import LinkMode
from grabber_pymunk.runner import GrabberRunner
from reacher_pymunk.runner import ReacherRunner

NAME = "test123"
DESCRIPTION = "Desc: Fixed links ((150 * 3) * 2), random square target, random start angles, curious, 256 batch\n" \
              "Network size: 4 * 32\n" \
              "DOF: 12 (2 (force, desired speed) x 6 seg)\n" \
              "Reward: 10 if target else 0\n" \
              "Granularity: 10"

grabber_runner = GrabberRunner(link_mode=LinkMode.FIXED, granularity=10, square_target=True)
grabber_runner.train(model_name=NAME, description=DESCRIPTION,
                     new_model=True, policy_hidden=[32, 32, 32, 32], curiosity_hidden=[32, 32, 32, 32],
                     checkpoint_interval=50000,
                     learning_rate=0.00025, gamma=0.95, batch_size=256, curiosity=True)
grabber_runner.demo(NAME)
