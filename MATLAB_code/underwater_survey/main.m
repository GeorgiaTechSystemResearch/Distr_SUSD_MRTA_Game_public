addpath('../utility_funcs/')
format shortG


N_HAUV = 3;
N_LAUV = 3;
N_agents = N_HAUV + N_LAUV; % total number of robots
N_tasks = 8; 
max_num_tasks = 5;% max number a robot can select
ymax = 5; % max number of subtasks in a task

env_type = 1; % 1 random with task spacing requirement (25m), 0 random without task spacing requirement
N_runs = 10;

folder = "results/"+datestr(now, 1)+ "_N"+num2str(N_agents)+"_L"+num2str(max_num_tasks)+ "_M"+num2str(N_tasks)+ "_ymax"+num2str(ymax)+"_N_HAUV"+num2str(N_HAUV) +"_env" +num2str(env_type)+"_grape_based";

mkdir(folder)

Q = ones(2,N_agents);
Q(1,1:N_HAUV) = 2;
Q(2,N_HAUV+1:N_agents) = 2;


N_timesteps = 200;
% LLL settings
n_selected = 500;

% agent
agent.id = 0; 
agent.loc = 0;                                                                  
agent.Q = 0;
agent.x = 0;
agent.theta = zeros(1, N_tasks*2);
agent.Theta = zeros(N_agents, N_tasks*2);
% agent.own_bids = zeros(1, N_tasks);
% agent.bidders = zeros(1, N_tasks);
agent.assignments = zeros(1, N_tasks);
agent.A = zeros(N_agents, N_tasks);
agent.utility = 0;
agent.selected_tasks = 0;
agent.max_num_tasks = max_num_tasks;
agent.path = [];
agent.agent_update_time = zeros(N_agents, 1);
agent.update_action_prob = 1;
agent.completion_time_vec = zeros(N_agents, 1);
agent.t_max = 0;   
agent.utility_func_type = 'concave';

if max_num_tasks == 1
    agent.MRTA_type = 'ST';
else 
    agent.MRTA_type = 'MT';
end

% Fully connected topology
A_fc = ones(N_agents);
for i = 1:N_agents
    A_fc(i, i) = 0;
end

% line topology
A_line= zeros(N_agents);
for i = 1:N_agents-1
    A_line(i, i+1) = 1;
    A_line(i+1, i) = 1;
end

% Loop topology
A_loop = zeros(N_agents);
for i = 1:N_agents
    k1 = mod((i-1),N_agents);
    if k1 == 0
        A_loop(i, N_agents) = 1;
    else 
        A_loop(i, k1) = 1;
    end

    k2 = mod((i+1),N_agents);
    if k2 == 0
        A_loop(i, N_agents) = 1;
    else 
        A_loop(i, k2) = 1;
    end
end

%% environment settings

map_size = [500;500]; 
env_rand_seeds = 1:N_runs;
N_envs = length(env_rand_seeds);

found_NE_vec_grape = zeros(1, N_envs);
tot_utility_vec_grape = zeros(1, N_envs);
update_time_grape = zeros(1, N_envs);
energy_consumption_vec_grape = zeros(1, N_envs);
completed_tasks_grape = zeros(1, N_envs);
total_utiliy_hist_grape = zeros(N_envs, N_timesteps);

found_NE_vec_hybrid = zeros(1, N_envs);
tot_utility_vec_hybrid = zeros(1, N_envs);
update_time_hybrid = zeros(1, N_envs);
energy_consumption_vec_hybrid = zeros(1, N_envs);
completed_tasks_hybrid = zeros(1, N_envs);
total_utiliy_hist_hybrid = zeros(N_envs, N_timesteps);


conflict_ratio_vec = zeros(1, N_envs);
% for N_tasks = [9, 12, 15, 18]
date_seed = datenum(datetime);
for env_i = 1:N_envs
    % Communication network
    A_comm = A_loop;
    G_comm = graph(A_comm);

    env_random_seed = env_rand_seeds(env_i);
    rng(env_random_seed + date_seed);
    subfolder = folder + "/rand"+num2str(env_random_seed);
    mkdir(subfolder)
    disp(['------------- Env', int2str(env_random_seed), '------------'])
    % robot locations
    % P_r = rand(2, N_agents).*map_size;
    P_r = zeros(2, N_agents);
    P_r(1,:) = -10;
    P_r(2,:) = (1:N_agents).*map_size(2)./N_agents.*0.8;
    % task locations
    % P_t = rand(2,N_tasks).*map_size;
        % define goal physical positions     
    if env_type == 1
        minimum_gap = 30;
        P_t = zeros(2,N_tasks);
        num_points_per_dim = floor(map_size/minimum_gap);
        loc_indices = randsample(num_points_per_dim(1)*num_points_per_dim(2), N_tasks);
        P_t(1,:) = mod(loc_indices, num_points_per_dim(1)).* minimum_gap;
        P_t(2,:) = fix(loc_indices/num_points_per_dim(2)).* minimum_gap;
        P_t = P_t + 5*(rand(size(P_t)));
    else
        P_t = rand(2,N_tasks).*map_size;
    end
    % task types
    Y = randi([1,ymax],2, N_tasks);
    null_tasks = find(Y(1,:)==0 & Y(2,:)==0);
    i = randi([1 2], 1, length(null_tasks));
    Y(i, null_tasks) = randi([1 ymax], length(null_tasks));
    %% GRAPE
    disp("GRAPE");
    agents = agent([]);     
    for agent_i = 1:N_agents
        agents(agent_i) = agent;
        agents(agent_i).id = agent_i;
        agents(agent_i).loc = P_r(:,agent_i);
        agents(agent_i).Q = Q;
        agents(agent_i).assignments = zeros(1, N_tasks);  % Initial task assignment: every robot is assigned to void task
    end 

    tic
    [agents_grape, output, t] = grape_MRTA_game_leqL(P_t, Y, agents, A_comm, N_timesteps); 
    toc
    total_utility = sum(output.a_utility);
    tic
    [reach_NE_grape, total_num_combs] = verify_reach_NE_para(agents_grape, P_t, Y);
    toc

    tot_energy_consumption_grape = cal_tot_energy_consumption(agents_grape, P_t, Y);

    paths = {};
    for agent_i = 1:N_agents
        paths{agent_i}  = agents_grape(agent_i).selected_tasks;
        % disp(max(0, agents(agent_i).theta));
    end
    selected_tasks = cell2mat(paths);
    [occurances, unique_tasks] = hist(selected_tasks,unique(selected_tasks));
    conflict_ratio = (length(find(occurances-1)))/length(unique_tasks);
    num_completed_tasks_grape = length(unique_tasks);

    disp([t, total_utility, num_completed_tasks_grape, tot_energy_consumption_grape, total_num_combs, reach_NE_grape])

    found_NE_vec_grape(1, env_i) = reach_NE_grape;
    tot_utility_vec_grape(1, env_i) = total_utility;
    update_time_grape(1, env_i) = t;
    energy_consumption_vec_grape(1, env_i) = tot_energy_consumption_grape;
    completed_tasks_grape(1, env_i) = num_completed_tasks_grape;
    utility_history_grape = output.utility_history;
    total_utiliy_hist_grape(env_i, :) = utility_history_grape(:,end)';

    % plot_utility_history(total_utiliy_hist_grape, 'GRAPE (game)', agents_lll)

    save(subfolder+'/agents_grape.mat','agents_grape');
    save(subfolder+'/utility_history_grape.mat','utility_history_grape');

    % paths = {};
    % for agent_i = 1:N_agents
    %     paths{agent_i}  = agents_grape(agent_i).path;
    %     % disp(max(0, agents(agent_i).theta));
    % end
    % fig = plot_paths(P_t, P_r, paths, "GRAPE U="+num2str(total_utility),Y);

    %% Hybrid (~ + SUSD)
    disp("Hybrid");
    ini_bias = 1.0;
    A = zeros(N_agents, N_tasks);
    agents_ini = agents_grape;
    % Syncranization
    for agent_i = 1:N_agents
        A(agent_i, :) = agents_ini(agent_i).assignments;
    end
    for agent_i = 1:N_agents
        agents_ini(agent_i).Theta(:, 1:N_tasks) = A*ini_bias;
        agents_ini(agent_i).Theta = agents_ini(agent_i).Theta + ones(size(agents_ini(agent_i).Theta))/10;
        agents_ini(agent_i).theta = agents_ini(agent_i).Theta(agent_i, :);
    end

    k1 = 0.6;
    k2 = 0.5;
    d_des = 0.3;
    N_susd_steps = 100;
    susd_hyperparams = [k1, k2, d_des, N_susd_steps];

    tic
    [agents_hybrid, total_utility, paths_history, utility_history_hybrid, t_h] = susd_MRTA_game_para(P_t, Y, agents_ini, G_comm, N_timesteps, susd_hyperparams);
    toc

    tic
    [reach_NE, total_num_combs] = verify_reach_NE_para(agents_hybrid, P_t, Y);
    toc

    tot_energy_consumption_hybrid = cal_tot_energy_consumption(agents_hybrid, P_t, Y);

    paths = {};
    for agent_i = 1:N_agents
        paths{agent_i}  = agents_hybrid(agent_i).selected_tasks;
        % disp(max(0, agents(agent_i).theta));
    end
    selected_tasks = cell2mat(paths);
    [occurances, unique_tasks] = hist(selected_tasks,unique(selected_tasks));
    conflict_ratio = (length(find(occurances-1)))/length(unique_tasks);
    num_completed_tasks_hybrid = length(unique_tasks);

    disp([t_h+t, total_utility, num_completed_tasks_hybrid, tot_energy_consumption_hybrid, total_num_combs, reach_NE])

    found_NE_vec_hybrid(1, env_i) = reach_NE;
    tot_utility_vec_hybrid(1, env_i) = total_utility;
    update_time_hybrid(1, env_i) = t_h+t;
    energy_consumption_vec_hybrid(1, env_i) = tot_energy_consumption_hybrid;
    completed_tasks_hybrid(1, env_i) = num_completed_tasks_hybrid;
    total_utiliy_hist_hybrid(env_i, :) = utility_history_hybrid(:,end)';

    % fig = plot_paths(P_t, P_r, paths, "Game theoretical U="+num2str(total_utility),Y);

    % plot_utility_history(utility_history_hybrid, 'Hybrid (game)', agents_hybrid)

    save(subfolder+'/agents_hybrid.mat','agents_hybrid');
    save(subfolder+'/hybrid_susd_hyperparams.mat','susd_hyperparams');
    save(subfolder+'/total_utiliy_hist_hybrid.mat','total_utiliy_hist_hybrid');

    %% Saving env info
    save(subfolder+'/P_r.mat','P_r');
    save(subfolder+'/P_t.mat','P_t');
    save(subfolder+'/Y.mat','Y');
    save(subfolder+'/Q.mat','Q');
end

disp([N_agents, N_tasks, max_num_tasks])


save(folder+'/found_NE_vec_grape.mat','found_NE_vec_grape');
save(folder+'/tot_utility_vec_grape.mat','tot_utility_vec_grape');
save(folder+'/update_time_grape.mat','update_time_grape');
save(folder+'/energy_consumption_vec_grape.mat','energy_consumption_vec_grape');
save(folder+'/completed_tasks_grape.mat','completed_tasks_grape');
save(folder+'/total_utiliy_hist_grape.mat','total_utiliy_hist_grape');

save(folder+'/found_NE_vec_hybrid.mat','found_NE_vec_hybrid');
save(folder+'/tot_utility_vec_hybrid.mat','tot_utility_vec_hybrid');
save(folder+'/update_time_hybrid.mat','update_time_hybrid');
save(folder+'/energy_consumption_vec_hybrid.mat','energy_consumption_vec_hybrid');
save(folder+'/completed_tasks_hybrid.mat','completed_tasks_hybrid');
save(folder+'/total_utiliy_hist_hybrid.mat','total_utiliy_hist_hybrid');

