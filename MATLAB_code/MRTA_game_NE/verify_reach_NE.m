function [reach_NE, total_num_combs] = verify_reach_NE(agents, P, Y)
    [~, N_agents] = size(agents);
    [~, N_tasks] = size(P);
    reach_NE = 1; 
    total_num_combs = 0;

    % syncranization
    for agent_i = 1:N_agents
        for agent_l = 1:N_agents
            agents(agent_i).A(agent_l,:) = agents(agent_l).A(agent_l,:);
        end
    end

    for agent_idx = 1:N_agents
        agent_i = agents(agent_idx); 
        org_utility = utility_calculation(agent_i, P, Y);
        max_num_tasks = agent_i.max_num_tasks;

        C = zeros(1, max_num_tasks);
        for number_selected_tasks = 1:max_num_tasks
            C1 = nchoosek(1:N_tasks, number_selected_tasks);
            C1(:, number_selected_tasks+1:max_num_tasks) = 0;
            C = [C;C1];
        end
        [n_actions, ~] = size(C);
        total_num_combs = total_num_combs + n_actions;  
        for a_l = 1:n_actions
            select_task_indices = C(a_l,:);
            select_task_indices = nonzeros(select_task_indices)';
            poss_agent = agent_i;
            assignments = zeros(1,N_tasks);
            assignments(select_task_indices) = 1;
            % poss_agent.theta = assignments;
            % poss_agent.Theta(agent_idx,:) = assignments;
            poss_agent.assignments = assignments;
            poss_agent.A(agent_idx,:) = assignments;
            poss_utility = utility_calculation(poss_agent, P, Y); 
            if any(agent_i.assignments ~= poss_agent.assignments) && poss_utility > org_utility
                disp([agent_idx, org_utility, poss_utility, (poss_utility-org_utility)/org_utility])
                disp(find(agent_i.assignments))
                disp(find(poss_agent.assignments))
                reach_NE = 0;
                break
            end
        end 
    end


    % switch agents(1).MRTA_type 
    %     case 'ST'
    %         for agent_idx = 1:N_agents
    %             agent_i = agents(agent_idx); 
    %             [org_assignment, org_utility] = utility_calculation(agent, P, Y)(agent_i, P, Y);  
    %             for task_idx = 1:N_tasks
    %                 poss_agent = agent_i;
    %                 assignments = zeros(1,N_tasks);
    %                 assignments(task_idx) = 1;
    %                 poss_agent.theta = assignments;
    %                 poss_agent.Theta(agent_idx,:) = assignments;
    %                 poss_agent.assignments = assignments;
    %                 poss_agent.A(agent_idx,:) = assignments;
    %                 [poss_assignment, poss_utility] = assignment_utility(poss_agent, P, Y);  
    %                 if any(org_assignment ~= poss_assignment) && poss_utility > org_utility
    %             	    fprintf('robot id, altertive assignemnt, altertive utility, utility differences \n');
    %                     disp([agent_idx, poss_assignment, poss_utility, (poss_utility-org_utility)/org_utility])
    %                     reach_NE = 0;
    %                     break
    %                 end 
    %             end
    %         end 
    %     otherwise
    %         for agent_idx = 1:N_agents
    %             agent_i = agents(agent_idx);        
    %             [org_assignment, org_utility] = assignment_utility(agent_i, P, Y);        
    %             C = nchoosek(1:N_tasks, agent_i.max_num_tasks);
    %             [n_actions, ~] = size(C);
    %             total_num_combs = total_num_combs + n_actions;         
    %             for a_l = 1:n_actions
    %                 select_task_indices = C(a_l,:);
    %                 poss_agent = agent_i;
    %                 assignments = zeros(1,N_tasks);
    %                 assignments(select_task_indices) = 1;
    %                 poss_agent.theta = assignments;
    %                 poss_agent.Theta(agent_idx,:) = assignments;
    %                 poss_agent.assignments = assignments;
    %                 poss_agent.A(agent_idx,:) = assignments;
    %                 [poss_assignment, poss_utility] = assignment_utility(poss_agent, P, Y); 
    %                 if any(org_assignment ~= poss_assignment) && poss_utility > org_utility
    %                     disp([agent_idx, tasks_order, poss_utility, (poss_utility-org_utility)/org_utility])
    %                     reach_NE = 0;
    %                     break
    %                 end
    %             end 
    %         end
    %     end
    % 

%     disp(total_num_combs)

end 