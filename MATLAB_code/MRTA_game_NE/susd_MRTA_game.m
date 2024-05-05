function [agents, total_utility, paths_history, utility_history, t] = susd_MRTA_game(P, Y, agents, G_comm, N_timesteps, hyperparams)
    gamma = 0.00;
    [~, N_agents] = size(agents);
    [~, N_tasks] = size(P);

    % % SUSD parameters (determinditic)
    % k1 = 0.5; % or 0.2
    % k2 = k1*0.5;
    % d_des = 0.5;

    k1 = hyperparams(1); %  0.5
    k2 = hyperparams(2); %  k1*0.5
    d_des = hyperparams(3); %  0.5;
    N_susd_steps = hyperparams(4);
    eps = 0.0;

    % % SUSD parameters (sampling)
    % k1 = 0.2;
    % k2 = k1*0.5;
    % d_des = 0.5;
    % eps = 0.0;
    % N_susd_steps = 60;


    paths_history = cell(N_timesteps, N_agents);
    utility_history = zeros(N_timesteps, N_agents+1);
    total_utility = 0;
    paths = [];

    P_r = zeros(2, N_agents);
    for a_i = 1:N_agents
        P_r(:,a_i) = agents(a_i).loc;
    end
    
    % Initilization:
    % for a_i = 1:N_agents
    %     agents(a_i).theta = rand(1, N_tasks);
    %     agents(a_i).Theta(a_i,:) = agents(a_i).theta;
    % end 
    
    %% Simulation Loop
    for t = 1:N_timesteps
        prev_agents = agents;
        % commmunicating robot's action (assignment)
        for a_i = 1:N_agents
            neighboring_agent_indices = neighbors(G_comm, a_i)'; % neighbors
            agents(a_i) = time_based_consensus_update(agents(a_i), prev_agents, neighboring_agent_indices);
            % agents(a_i) = leader_following_consensus_update(agents(a_i), prev_agents, neighboring_agent_indices, 0.2);
        end

        for a_i = 1:N_agents
            % % action update phase
            % if mod(t, 20) == 0  
            %     agents(a_i).theta = zeros(size(agents(a_i).theta));
            %     agents(a_i).theta(1, 1:N_tasks) = 0.5*agents(a_i).assignments;
            %     agents(a_i).theta = agents(a_i).theta + ones(size(agents(a_i).theta))*0.1;
            % end 
            p_update = agents(a_i).update_action_prob;
            % p_update = 1;
            update = randsample([0, 1], 1, true, [1-p_update, p_update]);
            % if mod(t,N_agents) == a_i-1
            %     update = 1;
            % end 
            if update == 1
                susd_hyperparams = [k1, k2, d_des, eps];
                prev_assigments = agents(a_i).assignments;
                agents(a_i) = action_update_susd(agents(a_i), P, Y, susd_hyperparams, N_susd_steps);
                if any(prev_assigments ~= agents(a_i).assignments)
                   agents(a_i).agent_update_time(a_i) = t;
                end
            end
        end
        
        %bookeeping
        for a_i = 1:N_agents
            poss_agent = agents(a_i);
            for a_l = 1:N_agents
                poss_agent.A(a_l,:) = agents(a_l).A(a_l,:);
            end
            agent_utility = utility_calculation(poss_agent, P, Y);
            utility_history(t, a_i) = agent_utility;
            utility_history(t, end) = utility_history(t, end) + agent_utility;
            paths_history(t, a_i) = {agents(a_i).path};
        end

        % early termination
        if max(extractfield(agents, 'agent_update_time')) + 20 < t
            for a_i = 1:N_agents
                utility_history(t+1:end, a_i) = utility_history(t, a_i);
            end 
            utility_history(t+1:end, end) = utility_history(t, end);
            break
        end

    end
    for a_i = 1:N_agents
        total_utility = total_utility + agents(a_i).utility;
    end
end