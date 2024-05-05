function assignment = theta_to_assignment(theta, max_num_tasks, converson_method)
    [~, theta_width] = size(theta); 
    num_tasks = theta_width/2;
    assignment = zeros(1, num_tasks);

    % converson_method = 'L_largest';
    switch converson_method
        
        case 'L_largest'
            % Deterministic approach 1 (L largest postive)
            theta = theta(1:num_tasks)-theta(num_tasks+1:theta_width);
            [~, select_task_indices] = maxk(theta.*(theta >= 0), max_num_tasks);
            assignment(select_task_indices) = 1;
            assignment = assignment.* (theta > 0);

        %% Deterministic approach 2 (postive elements in theta)
        % eps = 0.1;
        % assignment = zeros(1,N_tasks);
        % select_task_indices = (eps>0.1);
        % assignment(select_task_indices)=1;
        % TODO: penalty for exceeding max number of selected tasks

        case 'sampling' %sampling-based approach
            theta_pos = max(0, theta);
            poss_task_indices = find(theta_pos);
            if length(poss_task_indices) > max_num_tasks
                task_indices = datasample(1:num_tasks, max_num_tasks, 'Replace', false, 'Weights',theta_pos);
                % task_idx = randsample(1:num_tasks,1,true,max(0, theta));
                assignment(task_indices) = 1;
            else 
                assignment(poss_task_indices) = 1;
            end
        
        case 'randomized_rounding' % Randomized rounding
            theta = max(0, theta);
            theta = min(1, theta);
            poss_task_indices = find(theta);
            if length(poss_task_indices) <= max_num_tasks
                assignment(poss_task_indices) = 1;
                return
            end 
            [~,I] = sort(theta, 'descend');
            for j = 1:length(I)
                poss_task_idx = I(j);
                p = theta(poss_task_idx);
                add_task = randsample([0, 1], 1, true, [1-p, p]);
                if add_task == 1
                    assignment(poss_task_idx) = 1;
                end 
                if sum(assignment) >= max_num_tasks
                    break
                end
            end
    end
end