import gym

import deepr
import cartpole_continuing


def main():
    env = cartpole_continuing.CartPoleContinuingEnv()
    act = deepr.learn(
        env,
        network='mlp',
        method_type="shaping" #shielding, baseline
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
