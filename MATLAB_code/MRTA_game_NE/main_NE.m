clear all;
close all;

addpath('../utility_funcs/')
format shortG


N_tasks = 12;
N_agents = 4;
max_num_tasks = 6;
N_timesteps = 500;
% Q = ones(2, N_agents);

Q = [2,2,1,1;
     1,1,2,2];

% Q = [2,2,2,2,1,1,1,1;
%      1,1,1,1,2,2,2,2];

% if ~exist(folder, 'dir')
%     mkdir(folder)
%     folder_exist = 0;
% end
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
agent.utility_func_type = 'convex';
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
for comm_i = 2

if comm_i == 1
    folder = "results/"+datestr(now, 1)+ "_N"+num2str(N_agents)+"_L"+num2str(max_num_tasks)+ "_M"+num2str(N_tasks)+ '_U' + agent.utility_func_type +'_fc' + '_TimeBasedConsensus';
    folder_exist = 0;
    % Communication network
    A_comm = A_fc;
    G_comm = graph(A_comm);
else 
    folder = "results/"+datestr(now, 1)+ "_N"+num2str(N_agents)+"_L"+num2str(max_num_tasks)+ "_M"+num2str(N_tasks)+ '_U' + agent.utility_func_type + '_loop' + '_TimeBasedConsensus';
    folder_exist = 0;
    % Communication network
    A_comm = A_loop;
    G_comm = graph(A_comm);
end

map_size = [300;300]; 
env_rand_seeds = 1:5;
N_envs = length(env_rand_seeds);

if folder_exist == 0
    found_NE_vec_susd = zeros(1, N_envs);
    utility_vec_susd = zeros(1, N_envs);
    update_time_susd = zeros(1, N_envs);
    completed_tasks_susd = zeros(1, N_envs);
    total_utiliy_hist_susd = zeros(N_envs, N_timesteps);
    social_utility_vec_susd = zeros(1, N_envs);


    found_NE_vec_grape = zeros(1, N_envs);
    utility_vec_grape = zeros(1, N_envs);
    update_time_grape = zeros(1, N_envs);
    completed_tasks_grape = zeros(1, N_envs);
    total_utiliy_hist_grape = zeros(N_envs, N_timesteps);
    social_utility_vec_grape = zeros(1, N_envs);

    found_NE_vec_hybrid = zeros(1, N_envs);
    utility_vec_hybrid = zeros(1, N_envs);
    update_time_hybrid = zeros(1, N_envs);
    completed_tasks_hybrid = zeros(1, N_envs);
    total_utiliy_hist_hybrid = zeros(N_envs, N_timesteps);
    social_utility_vec_hybrid = zeros(1, N_envs);       
else
    load(folder+'/found_NE_vec_susd.mat');
    load(folder+'/utility_vec_susd.mat');
    load(folder+'/update_time_susd.mat');
    load(folder+'/completed_tasks_susd.mat');
    load(folder+'/total_utiliy_hist_susd.mat');
    load(folder+'/social_utility_vec_susd.mat');

    load(folder+'/found_NE_vec_grape.mat');
    load(folder+'/utility_vec_grape.mat');
    load(folder+'/update_time_grape.mat');
    load(folder+'/completed_tasks_grape.mat');
    load(folder+'/total_utiliy_hist_grape.mat');
    load(folder+'/social_utility_vec_grape.mat');
end 

conflict_ratio_vec = zeros(1, N_envs);
date_seed = datenum(datetime);
for e_idx = 1:N_envs
    env_id = env_rand_seeds(e_idx);
    env_random_seed = env_rand_seeds(e_idx);
    rng(env_random_seed + date_seed);
    subfolder = folder + "/rand"+num2str(env_id);
    mkdir(subfolder)
    disp(['------------- Env', int2str(env_id), '------------'])
    % robot locations
    %x0 = [0 0; 3 5; 5 0; 0 5; 5 5]';
    P_r = rand(2, N_agents).*map_size;
    
    % task locations
    P_t = rand(2,N_tasks).*map_size;
    % task types 
    indices = randi([1 2],1,N_tasks);
    Y = zeros(2, N_tasks);
    for i=1:N_tasks
        Y(indices(i),i) = 1;
    end
    % Y = ones(2, N_tasks);
    
    % Game start 
    
    %% SUSD 
    disp("SUSD");
    % susd hyperparameters
    k1 = 0.3;
    k2 = 0.2;
    d_des = 0.2;
    N_susd_steps = 100;
    susd_hyperparams = [k1, k2, d_des, N_susd_steps];

    agents = agent([]);     
    for agent_i = 1:N_agents
        agents(agent_i) = agent;
        agents(agent_i).id = agent_i;
        agents(agent_i).loc = P_r(:,agent_i);
        agents(agent_i).Q = Q;
    end 


    tic
    [agents_susd, total_utility, paths_history_susd, utility_history_susd, t] = susd_MRTA_game(P_t, Y, agents, G_comm, N_timesteps, susd_hyperparams);
    toc

    tic
    [reach_NE, total_num_combs] = verify_reach_NE(agents_susd, P_t, Y);
    toc

    socical_utility_susd = cal_socical_utility(agents_susd, P_t, Y);
    disp([t, total_utility, total_num_combs, reach_NE])


    found_NE_vec_susd(1, env_id) = reach_NE;
    utility_vec_susd(1, env_id) = total_utility;
    update_time_susd(1, env_id) = t;
    % completed_tasks_susd(1, env_id) = num_completed_tasks_susd;
    total_utiliy_hist_susd(env_id, :) = utility_history_susd(:,end)';
    social_utility_vec_susd(1, env_id) = socical_utility_susd;

    for agent_i = 1:N_agents
        paths{agent_i}  = agents_susd(agent_i).path;
        % disp(max(0, agents(agent_i).theta));
    end

    fig = plot_paths(P_t, P_r, paths, "Game theoretical U="+num2str(total_utility), Y);

    selected_tasks = cell2mat(paths);
    [occurances, unique_tasks] = hist(selected_tasks,unique(selected_tasks));
    conflict_ratio = (length(find(occurances-1)))/length(unique_tasks);
    % disp([unique_tasks, occurances, conflict_ratio]);

    % saveas(fig, subfolder +'/paths.png');
    save(subfolder+'/agents_susd.mat','agents_susd');
    save(subfolder+'/utility_history_susd.mat','utility_history_susd');
    save(subfolder+'/susd_hyperparams.mat','susd_hyperparams');


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
    reach_NE_grape = 0;
    total_num_combs = 0;
    [reach_NE_grape, total_num_combs] = verify_reach_NE(agents_grape, P_t, Y);
    toc
    % [makespan_grape, num_completed_tasks_grape] = makespan_calculation(agents_grape, P_t, Y);

    socical_utility_grape = cal_socical_utility(agents_grape, P_t, Y);

    disp([t, total_utility, total_num_combs, reach_NE_grape])

    found_NE_vec_grape(1, env_id) = reach_NE_grape;
    utility_vec_grape(1, env_id) = total_utility;
    update_time_grape(1, env_id) = t;
    % completed_tasks_grape(1, env_id) = num_completed_tasks_grape;
    utility_history_grape = output.utility_history;
    total_utiliy_hist_grape(env_id, :) = utility_history_grape(:,end)';
    social_utility_vec_grape(1, env_id) = socical_utility_grape;

    % plot_utility_history(total_utiliy_hist_grape, 'GRAPE (game)', agents_lll)

    save(subfolder+'/agents_grape.mat','agents_grape');
    save(subfolder+'/utility_history_grape.mat','utility_history_grape');

    %% Hybrid (~ + SUSD)
    disp("Hybrid");
    ini_bias = 3.0;
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

    k1 = 0.3; %1.6 or 0.3
    k2 = 0.2; %0.8 or 0.2
    d_des = 0.2;
    N_susd_steps = 50;
    susd_hyperparams = [k1, k2, d_des, N_susd_steps];

    tic
    [agents_hybrid, total_utility, paths_history, utility_history_hybrid, t_h] = susd_MRTA_game(P_t, Y, agents_ini, G_comm, N_timesteps, susd_hyperparams);
    toc

    tic
    [reach_NE, total_num_combs] = verify_reach_NE(agents_hybrid, P_t, Y);
    toc

    socical_utility_hybrid = cal_socical_utility(agents_hybrid, P_t, Y);

    paths = {};
    for agent_i = 1:N_agents
        paths{agent_i}  = agents_hybrid(agent_i).selected_tasks;
        % disp(max(0, agents(agent_i).theta));
    end
    selected_tasks = cell2mat(paths);
    [occurances, unique_tasks] = hist(selected_tasks,unique(selected_tasks));
    conflict_ratio = (length(find(occurances-1)))/length(unique_tasks);
    num_completed_tasks_hybrid = length(unique_tasks);

    disp([t_h+t, total_utility, socical_utility_hybrid, total_num_combs, reach_NE])

    found_NE_vec_hybrid(1, env_id) = reach_NE;
    utility_vec_hybrid(1, env_id) = total_utility;
    social_utility_vec_hybrid(1, env_id) = socical_utility_hybrid;
    update_time_hybrid(1, env_id) = t_h+t;
    completed_tasks_hybrid(1, env_id) = num_completed_tasks_hybrid;
    total_utiliy_hist_hybrid(env_id, :) = utility_history_hybrid(:,end)';

    paths = {};
    for agent_i = 1:N_agents
        paths{agent_i}  = agents_hybrid(agent_i).path;
        % disp(max(0, agents(agent_i).theta));
    end
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
save(folder+'/A_comm.mat','A_comm');

save(folder+'/found_NE_vec_susd.mat','found_NE_vec_susd');
save(folder+'/utility_vec_susd.mat','utility_vec_susd');
save(folder+'/update_time_susd.mat','update_time_susd');
save(folder+'/completed_tasks_susd.mat','completed_tasks_susd');
save(folder+'/total_utiliy_hist_susd.mat','total_utiliy_hist_susd');
save(folder+'/social_utility_vec_susd.mat','social_utility_vec_susd');

save(folder+'/found_NE_vec_grape.mat','found_NE_vec_grape');
save(folder+'/utility_vec_grape.mat','utility_vec_grape');
save(folder+'/update_time_grape.mat','update_time_grape');
save(folder+'/completed_tasks_grape.mat','completed_tasks_grape');
save(folder+'/total_utiliy_hist_grape.mat','total_utiliy_hist_grape');
save(folder+'/social_utility_vec_grape.mat','social_utility_vec_grape');

save(folder+'/found_NE_vec_hybrid.mat','found_NE_vec_hybrid');
save(folder+'/utility_vec_hybrid.mat','utility_vec_hybrid');
save(folder+'/update_time_hybrid.mat','update_time_hybrid');
save(folder+'/completed_tasks_hybrid.mat','completed_tasks_hybrid');
save(folder+'/total_utiliy_hist_hybrid.mat','total_utiliy_hist_hybrid');
save(folder+'/social_utility_vec_hybrid.mat','social_utility_vec_hybrid');


end