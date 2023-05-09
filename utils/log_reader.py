import json, sys, os

if __name__ == '__main__':

    log_dir = sys.argv[1]
    key = sys.argv[2]

    run_names = os.listdir(log_dir)
    run_names.sort()

    for run in run_names:
        with open(os.path.join(log_dir, run), 'r') as f:
            for line in f:
                if '[LOG_READER]' in line:
                    data = json.loads(line.strip().replace('[LOG_READER]', ''))
                    lang = run.replace('.log', '')
                    print('{},{}'.format(lang, data[key]))



