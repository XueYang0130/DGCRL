import numpy as np
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


# print("abc"+str(len(np.load("result/DGTD3/hopper/reference/DGCRL_returns_hopper_reference_14.npy")[0])))
def get_forgetting(performance_T, performances):
    performance_i_Ms = []
    for entry in performances:
        # print(entry)
        performance_i_Ms.append(entry[-1])
    forgettings = []
    for i in range(len(performance_i_Ms)):
        forgetting_i = performance_i_Ms[i] - performance_T[i]
        forgettings.append(forgetting_i)
    print(f"Forgetting is: {np.mean(forgettings)}")
    return np.mean(forgettings)


def calculate_area(x, y, c):
    assert len(x) == len(y), "x and y must have the same length"

    area = 0.0
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        y_avg = (y[i] + y[i - 1]) / 2.0
        area += dx * abs(y_avg - c)
    return area


forgettings = []
forward_transfers = []
pTs = []
avg_return_epss = []
return_alls = []
baseline = "DGTD3"
analysis = ""
sensitivity_analysis = False
# ablation_study= True
env_name = "navigation_v2"
demo_num = 30
num_episodes = 100
for seed in [
    6,
    14,
    20,
             202,
             405
             ]:
    # return_eps = np.load("result/" + baseline + "/" + env_name + "/_" + str(seed) + ".npy")
    # avg_return_eps = np.mean(return_eps, axis=0)
    # avg_return_epss.append(avg_return_eps)
    #
    # return_all = np.mean(return_eps)
    # return_alls.append(return_all)
    # # print(np.load("result/LLIRL/half_cheetah/rews_llirl_"+str(seed)+".npy"))
    #
    # # performance_T = np.load("result/" + baseline + "/" + env_name + "/T_" + str(seed) + ".npy")[:, -1]
    # performance_T = np.load("result/" + baseline + "/" + env_name + "/T_" + str(seed) + ".npy")
    # print(performance_T)
    # performances = np.load("result/" + baseline + "/" + env_name + "/_" + str(seed) + ".npy")
    # forgetting = get_forgetting(performance_T=performance_T, performances=performances)
    # forgettings.append(forgetting)
    # pTs.append(np.mean(performance_T))
    # # print(np.load("result/DGTD3/half_cheetah/reference/DGCRL_returns_half_cheetah_reference_6.npy"))
    # # print(np.min(np.load("result/DGTD3/half_cheetah/reference/DGCRL_returns_half_cheetah_reference_6.npy")))
    # #
    # x = range(num_episodes)
    y_ref = np.load(
        "result/DGTD3/" + env_name + "/reference/DGCRL_returns_" + env_name + "_reference_" + str(seed) + ".npy")
    # y_DGCRL = np.load("result/DGTD3/half_cheetah/DGCRL_returns_half_cheetah_6.npy")
    y = np.load("result/" + baseline + "/" + env_name + "/_" + str(seed) + ".npy")
    # if sensitivity_analysis:
    return_eps = np.load("result/" + baseline + "/sensitivity_analysis/demo_num/"+ env_name + f"/{demo_num}/_" + str(seed) + ".npy")
    # return_eps = np.load(
    #     "result/" + baseline + "/ablation_study/" + env_name + f"/actor/_" + str(seed) + ".npy")
    avg_return_eps = np.mean(return_eps, axis=0)
    avg_return_epss.append(avg_return_eps)

    return_all = np.mean(return_eps)
    return_alls.append(return_all)
        # print(np.load("result/LLIRL/half_cheetah/rews_llirl_"+str(seed)+".npy"))

        # performance_T = np.load("result/" + baseline + "/" + env_name + "/T_" + str(seed) + ".npy")[:, -1]
    performance_T = np.load("result/" + baseline + "/sensitivity_analysis/demo_num/" + env_name + f"/{demo_num}/T_" + str(seed) + ".npy")
    performances = np.load("result/" + baseline + "/sensitivity_analysis/demo_num/" + env_name + f"/{demo_num}/_" + str(seed) + ".npy")

    # performance_T = np.load(
    #     "result/" + baseline + "/ablation_study/" + env_name + f"/actor/T_" + str(seed) + ".npy")
    # print("0------------------------------------")
    # print(np.load(
    #     "result/" + baseline + "/ablation_study/" + env_name + f"/actor/T_" + str(seed) + ".npy"))
    # print(np.load(
    #     "result/" + baseline + "/ablation_study/" + env_name + f"/critic/T_" + str(seed) + ".npy"))
    # print(np.load(
    #     "result/" + baseline + "/ablation_study/" + env_name + f"/actor_critic/T_" + str(seed) + ".npy"))
    # print("0------------------------------------")
    # performances = np.load(
    #     "result/" + baseline + "/ablation_study/" + env_name + f"/actor/_" + str(seed) + ".npy")
    print(performances)
    print(performance_T)
    forgetting = get_forgetting(performance_T=performance_T, performances=performances)
    forgettings.append(forgetting)
    pTs.append(np.mean(performance_T))
        # print(np.load("result/DGTD3/half_cheetah/reference/DGCRL_returns_half_cheetah_reference_6.npy"))
        # print(np.min(np.load("result/DGTD3/half_cheetah/reference/DGCRL_returns_half_cheetah_reference_6.npy")))
        #
    y = np.load("result/" + baseline + "/sensitivity_analysis/demo_num/" + env_name + f"/{demo_num}/_" + str(seed) + ".npy")
    # y = np.load(
    #     "result/" + baseline + "/ablation_study/" + env_name + f"/actor/_" + str(seed) + ".npy")
    x = range(num_episodes)
    c = np.min([np.min(y), np.min(y_ref)])
    max_v = np.max([np.max(y), np.max(y_ref)])

    AUC_DGCRL = 0
    AUC_ref = 0
    FT = 0
    max_area = calculate_area(x, np.ones(len(x)) * max_v, c) / num_episodes
    for i in range(len(y_ref)):
        AUC_i_DGCRL = calculate_area(x, y[i], c) / num_episodes
        AUC_i_ref = calculate_area(x, y_ref[i], c) / num_episodes
        AUC_DGCRL += AUC_i_DGCRL
        AUC_ref += AUC_i_ref

        FT_i = (AUC_i_DGCRL - AUC_i_ref) / (max_area - AUC_i_ref)
        FT += FT_i
    forward_transfers.append(FT / 50)
print(f"Results for :{env_name} with baseline:{baseline}")
print(f"final performances:{pTs}\nforward_transfers:{forward_transfers}\nforgetting:{forgettings}\n"
      # f"avg_return_epss:{avg_return_epss}\nreturn_alls:{return_alls}"
      )
# np.save("result/statistics_/" + baseline + "/" + env_name + "/pTs.npy", pTs)
# np.save("result/statistics_/" + baseline + "/" + env_name + "/forward_transfers.npy", forward_transfers)
# np.save("result/statistics_/" + baseline + "/" + env_name + "/forgettings.npy", forgettings)
# np.save("result/statistics_/" + baseline + "/" + env_name + "/avg_return_epss.npy", avg_return_epss)
# np.save("result/statistics_/" + baseline + "/" + env_name + "/return_alls.npy", return_alls)

# np.save("result/statistics_/" + baseline + "/ablation_study/" + env_name + f"/actor/pTs.npy", pTs)
# np.save("result/statistics_/" + baseline + "/ablation_study/" + env_name + f"/actor/forward_transfers.npy", forward_transfers)
# np.save("result/statistics_/" + baseline + "/ablation_study/" + env_name + f"/actor/forgettings.npy", forgettings)
# np.save("result/statistics_/" + baseline + "/ablation_study/" + env_name + f"/actor/avg_return_epss.npy", avg_return_epss)
# np.save("result/statistics_/" + baseline + "/ablation_study/" + env_name + f"/actor/return_alls.npy", return_alls)

np.save("result/statistics_/" + baseline + "/sensitivity_analysis/demo_num/" + env_name + f"/{str(demo_num)}/pTs.npy", pTs)
np.save("result/statistics_/" + baseline + "/sensitivity_analysis/demo_num/" + env_name + f"/{str(demo_num)}/forward_transfers.npy", forward_transfers)
np.save("result/statistics_/" + baseline + "/sensitivity_analysis/demo_num/" + env_name + f"/{str(demo_num)}/forgettings.npy", forgettings)
np.save("result/statistics_/" + baseline + "/sensitivity_analysis/demo_num/" + env_name + f"/{str(demo_num)}/avg_return_epss.npy", avg_return_epss)
np.save("result/statistics_/" + baseline + "/sensitivity_analysis/demo_num/" + env_name + f"/{str(demo_num)}/return_alls.npy", return_alls)