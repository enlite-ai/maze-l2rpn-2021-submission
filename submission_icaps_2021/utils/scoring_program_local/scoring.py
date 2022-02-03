import os
import shutil
import io
import sys
import json
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import glob

SCORE_TXT = "scores.txt"
ALL_SCORE_CSV = "score_episodes.csv"
RESULT_HTML = "results.html"
META_JSON = "episode_meta.json"
TIME_JSON = "episode_times.json"
REWARD_JSON = "other_rewards.json"
SCORES_JSON = "res_agent.json"
MIN_SCORE = -130.


def create_fig(title, x=4, y=2, width=1280, height=720, dpi=96):
    w = width / dpi
    h = height / dpi
    fig, axs = plt.subplots(ncols=x, nrows=y, figsize=(w, h), sharey=True)
    fig.suptitle(title)
    return fig, axs


def draw_steps_fig(ax, step_data, ep_score, ep_share):
    nb_timestep_played = int(step_data["nb_timestep_played"])
    chronics_max_timestep = int(step_data["chronics_max_timestep"])
    n_blackout_steps = chronics_max_timestep - nb_timestep_played
    title_fmt = "Scenario {}\n({:.2f}/{:.2f})"
    scenario_dir = os.path.basename(step_data["chronics_path"])
    scenario_name = title_fmt.format(scenario_dir, ep_score, ep_share)
    labels = 'Played', 'Blackout'
    colors = ['blue', 'orange']
    fracs = [
        nb_timestep_played,
        n_blackout_steps
    ]

    def pct_fn(pct):
        n_steps = int(pct * 0.01 * chronics_max_timestep)
        return "{:.1f}%\n({:d})".format(pct, n_steps)

    ax.pie(fracs, labels=labels,
           autopct=pct_fn,
           startangle=90.0)
    ax.set_title(scenario_name)


def draw_rewards_fig(ax, reward_data, step_data):
    nb_timestep_played = int(step_data["nb_timestep_played"])
    chronics_max_timestep = int(step_data["chronics_max_timestep"])
    n_blackout_steps = chronics_max_timestep - nb_timestep_played
    scenario_name = "Scenario " + os.path.basename(step_data["chronics_path"])

    x = list(range(chronics_max_timestep))
    n_rewards = len(reward_data[0].keys())
    y = [[] for _ in range(n_rewards)]
    labels = list(reward_data[0].keys())
    for rel in reward_data:
        for i, v in enumerate(rel.values()):
            y[i].append(v)
    for i in range(n_rewards):
        y[i] += [0.0] * n_blackout_steps

    for i in range(n_rewards):
        ax.plot(x, y[i], label=labels[i])
    ax.set_title(scenario_name)
    ax.legend()


def fig_to_b64(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    fig_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return fig_b64


def html_result(score, duration, op_score, att_score, fig_list):
    html = """<html><head></head><body>\n"""
    html += """<div style='margin: 0 auto; width: 500px;'>"""
    html += """<p>Score {}</p>""".format(np.round(score, 3))
    html += """<p>Duration {}</p>""".format(np.round(duration, 2))
    html += """<p>Operational Score {}</p>""".format(np.round(op_score, 2))
    html += """<p>Attention Score {}</p>""".format(np.round(att_score, 2))
    html += """</div>"""
    for i, figure in enumerate(fig_list):
        html += '<img src="data:image/png;base64,{0}"><br>'.format(figure)
    html += """</body></html>"""
    return html


def html_error():
    html = """<html><head></head><body>\n"""
    html += """Invalid submission"""
    html += """</body></html>"""
    return html


def cli():
    DEFAULT_TIMEOUT_SECONDS = 20*60
    DEFAULT_NB_EPISODE = 10
    DEFAULT_KEY_SCORE = "tmp_score_codalab"
    DEFAULT_Complete_Score_Output = True  # False #do we log each score per scenario in .txt or only total score

    parser = argparse.ArgumentParser(description="Scoring program")
    parser.add_argument("--logs_in", required=True,
                        help="Path to the runned output directory")
    parser.add_argument("--config_in", required=True,
                        help="DoNothing json config input file")
    parser.add_argument("--data_out", required=True,
                        help="Path to the results output directory")
    parser.add_argument("--key_score", required=False,
                        default=DEFAULT_KEY_SCORE, type=str,
                        help="Codalab other_reward name")
    parser.add_argument("--timeout_seconds", required=False,
                        default=DEFAULT_TIMEOUT_SECONDS, type=int,
                        help="Number of seconds before codalab timeouts")
    parser.add_argument("--nb_episode", required=False,
                        default=DEFAULT_NB_EPISODE, type=int,
                        help="Number of episodes in logs in")
    
    parser.add_argument("--all_score", required=False,
                        default=DEFAULT_Complete_Score_Output, type=bool,
                        help="save each score per scenario")
    
    return parser.parse_args()


def write_output(output_dir, html_content, episode_score_dic, duration,save_all_score=False):
    # Make sure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Write scores
    score_filename = os.path.join(output_dir, SCORE_TXT)
    with open(score_filename, 'w') as f:
        f.write("score: {:.6f}\n".format(episode_score_dic["total"]))
        f.write("duration: {:.6f}\n".format(duration))
        f.write("total_operation: {:.6f}\n".format(episode_score_dic["total_operation"]))
        f.write("total_attention: {:.6f}\n".format(episode_score_dic["total_attention"]))

    if save_all_score:
        print(f"\t\t saving all scores")
        all_score_filename = os.path.join(output_dir, ALL_SCORE_CSV)
        global_keys = {"total", "total_operation", "total_attention"}
        episode_score_df = {'scenario': [k for k, val in episode_score_dic.items() if k not in global_keys],
                            'score': [val[0] for k, val in episode_score_dic.items() if k not in global_keys],
                            "operation_score": [val[1] for k, val in episode_score_dic.items() if k not in global_keys],
                            "attention_score": [val[2] for k, val in episode_score_dic.items()if k not in global_keys]}
        episode_score_df["scenario"].append("global")
        episode_score_df["score"].append(episode_score_dic["total"])
        episode_score_df["operation_score"].append(episode_score_dic["total_operation"])
        episode_score_df["attention_score"].append(episode_score_dic["total_attention"])
        all_score_pd = pd.DataFrame(episode_score_df).round(1)
        all_score_pd.to_csv(all_score_filename, index=False)

    # Write results
    result_filename = os.path.join(output_dir, RESULT_HTML)
    with open(result_filename, 'w') as f:
        f.write(html_content)


def main():
    args = cli()
    input_dir = args.logs_in
    output_dir = args.data_out
    config_file = args.config_in
    save_all_score = args.all_score

    print("\t\t input dir: {}".format(input_dir))
    print("\t\t output dir: {}".format(output_dir))
    print("\t\t config json: {}".format(config_file))
    print("\t\t input content", os.listdir(input_dir))

    with open(config_file, "r") as f:
        config = json.load(f)

    # Fail if input doesn't exists
    if not os.path.exists(input_dir):
        error_score = (MIN_SCORE, -100., -200.)
        error_duration = args.timeout_seconds + 1
        write_output(output_dir, html_error(), error_score, error_duration)
        sys.exit("Your submission is not valid.")

    # Create output variables
    total_duration = 0.0
    total_score = 0.0
    total_operational_score = 0.0
    total_attention_score = 0.0
    ## Create output figures
    step_w = 4
    step_h = max(args.nb_episode // step_w, 1)
    if args.nb_episode % step_w > 0:
        step_h += 1
    step_fig, step_axs = create_fig("Completion", x=step_w, y=step_h)
    reward_w = 2
    reward_h = max(args.nb_episode // reward_w, 1)
    reward_title = "Cost of grid operation & Custom rewards"
    reward_fig, reward_axs = create_fig(reward_title,
                                        x=reward_w, y=reward_h,
                                        height=1750)
    episode_index = 0
    episode_names = config["episodes_info"].keys()
    score_config = config["score_config"]

    episode_score_dic = {}
    scores_json = os.path.join(input_dir, SCORES_JSON)
    if not os.path.exists(scores_json):
        # the score has not been computed
        total_duration = 99999
        total_score = MIN_SCORE
        total_operational_score = -100.
        total_attention_score = -200.
    else:
        with open(scores_json, "r", encoding="utf-8") as f:
            dict_scores = json.load(fp=f)
        # I loop through all the episodes
        for episode_id in sorted(episode_names):
            # Get info from config
            episode_info = config["episodes_info"][episode_id]
            episode_len = float(episode_info["length"])
            episode_weight = episode_len / float(score_config["total_timesteps"])

            # Compute episode files paths
            scenario_dir = os.path.join(input_dir, episode_id)
            meta_json = os.path.join(scenario_dir, META_JSON)
            time_json = os.path.join(scenario_dir, TIME_JSON)
            reward_json = os.path.join(scenario_dir, REWARD_JSON)
            if not os.path.isdir(scenario_dir) or \
               not os.path.exists(meta_json) or \
               not os.path.exists(time_json) or \
               not os.path.exists(reward_json) or \
               not os.path.exists(scores_json):
                episode_score = MIN_SCORE
            else:
                with open(meta_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                with open(reward_json, "r", encoding="utf-8") as f:
                    other_rewards = json.load(f)
                with open(time_json, "r", encoding="utf-8") as f:
                    timings = json.load(f)

                episode_score = dict_scores["scores"][episode_index][0]  # for the total score
                episode_score_dic[episode_id] = dict_scores["scores"][episode_index]  # for all the scores

                # Draw figs
                step_ax_x = episode_index % step_w
                step_ax_y = episode_index // step_w
                draw_steps_fig(step_axs[step_ax_y, step_ax_x],
                               meta, episode_score * episode_weight,
                               episode_weight * 100.0)
                reward_ax_x = episode_index % reward_w
                reward_ax_y = episode_index // reward_w
                draw_rewards_fig(reward_axs[reward_ax_y, reward_ax_x],
                                 other_rewards, meta)

            # Sum durations and scores
            total_duration += float(timings["Agent"]["total"])
            total_score += episode_weight * episode_score
            total_operational_score += episode_weight * float(dict_scores["scores"][episode_index][1])
            total_attention_score += episode_weight * float(dict_scores["scores"][episode_index][2])

            # Loop to next episode
            episode_index += 1

    episode_score_dic["total"] = total_score
    episode_score_dic["total_operation"] = total_operational_score
    episode_score_dic["total_attention"] = total_attention_score

    # Format result html page
    step_figb64 = fig_to_b64(step_fig)
    reward_figb64 = fig_to_b64(reward_fig)
    html_out = html_result(total_score, total_duration,
                           op_score=total_operational_score,
                           att_score=total_attention_score,
                           fig_list=[step_figb64, reward_figb64])

    # Write final output
    print(f"\t\t: {episode_score_dic}")
    write_output(output_dir, html_out, episode_score_dic, total_duration,save_all_score)
    try:
        # Copy gifs if any
        gif_input = os.path.abspath(input_dir)
        gif_output = os.path.abspath(output_dir)
        gif_names = glob.glob(os.path.join(gif_input, "*.gif"))
        if gif_names:
            for g_n in gif_names:
                g_n_local = os.path.split(g_n)[-1]
                shutil.copy(os.path.join(g_n),
                            os.path.join(gif_output, g_n_local))
        # gif_cmd = "find {} -name '*.gif' | xargs -i cp {} {}"
        # os.system(gif_cmd.format(gif_input, "{}", gif_output))
    except Exception as exc_:
        print(f"\t\t WARNING: GIF copy failed, no gif will be available. Error was: {exc_}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as exc_:
        print("------------------------------------")
        print("        Detailed error Logs         ")
        print("------------------------------------")
        print("ERROR: scoring program failed with error: \n{}".format(exc_))
        print("Traceback is:\n")
        traceback.print_exc(file=sys.stdout)
        print("------------------------------------")
        print("        Detailed error Logs         ")
        print("------------------------------------")
        sys.exit(1)
