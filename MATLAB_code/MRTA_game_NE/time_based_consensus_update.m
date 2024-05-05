function agent_self = time_based_consensus_update(agent, prev_agents, neighboring_agent_indices)
    beta = 0.5;
    prev_A = agent.A;
    [~, N_agents] = size(prev_agents);
    % communication phase (Time-based update consensus)
    for a_k = neighboring_agent_indices % neighbors
        for a_l = 1:N_agents
            if a_l ~= agent.id
                if prev_agents(a_k).agent_update_time(a_l) > agent.agent_update_time(a_l)
                    agent.A(a_l,:) = prev_agents(a_k).A(a_l,:);
                    agent.agent_update_time(a_l) = prev_agents(a_k).agent_update_time(a_l);
                end
            end
        end
    end
    
    % agent.update_action_prob = 1;
    agent.update_action_prob = min(exp(-beta*norm(max(0,agent.A)-max(0,prev_A))), 1);

    agent_self = agent;