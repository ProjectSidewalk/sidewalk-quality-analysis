-- The big query with most of the data! Output as ml-<city-name>-stats.csv
SELECT users.user_id,
    CASE WHEN role.role = 'Anonymous' THEN 'Anonymous' WHEN role.role = 'Registered' THEN 'Registered' WHEN role.role = 'Turker' THEN 'Turker' ELSE 'Researcher' END AS role,
    n_validation_received,
    n_received_curb_ramp_agree + n_received_missing_curb_ramp_agree + n_received_obstacle_agree + n_received_surface_problem_agree AS n_received_agree,
    n_received_curb_ramp_disagree + n_received_missing_curb_ramp_disagree + n_received_obstacle_disagree + n_received_surface_problem_disagree AS n_received_disagree,
    n_received_curb_ramp_unsure + n_received_missing_curb_ramp_unsure + n_received_obstacle_unsure + n_received_surface_problem_unsure AS n_received_unsure,
    n_received_curb_ramp_tie + n_received_missing_curb_ramp_tie + n_received_obstacle_tie + n_received_surface_problem_tie AS n_received_tie,
    n_received_curb_ramp_agree + n_received_curb_ramp_disagree + n_received_curb_ramp_unsure + n_received_curb_ramp_tie AS n_received_curb_ramp,
    n_received_curb_ramp_agree,
    n_received_curb_ramp_disagree,
    n_received_curb_ramp_unsure,
    n_received_curb_ramp_tie,
    n_received_missing_curb_ramp_agree + n_received_missing_curb_ramp_disagree + n_received_missing_curb_ramp_unsure + n_received_missing_curb_ramp_tie AS n_received_missing_curb_ramp,
    n_received_missing_curb_ramp_agree,
    n_received_missing_curb_ramp_disagree,
    n_received_missing_curb_ramp_unsure,
    n_received_missing_curb_ramp_tie,
    n_received_obstacle_agree + n_received_obstacle_disagree + n_received_obstacle_unsure + n_received_obstacle_tie AS n_received_obstacle,
    n_received_obstacle_agree,
    n_received_obstacle_disagree,
    n_received_obstacle_unsure,
    n_received_obstacle_tie,
    n_received_surface_problem_agree + n_received_surface_problem_disagree + n_received_surface_problem_unsure + n_received_surface_problem_tie AS n_received_surface_problem,
    n_received_surface_problem_agree,
    n_received_surface_problem_disagree,
    n_received_surface_problem_unsure,
    n_received_surface_problem_tie,
    COALESCE(n_validation_given, 0) AS n_validation_given,
    COALESCE(n_given_curb_ramp_agree + n_given_missing_curb_ramp_agree + n_given_obstacle_agree + n_given_surface_problem_agree, 0) AS n_given_agree,
    COALESCE(n_given_curb_ramp_disagree + n_given_missing_curb_ramp_disagree + n_given_obstacle_disagree + n_given_surface_problem_disagree, 0) AS n_given_disagree,
    COALESCE(n_given_curb_ramp_unsure + n_given_missing_curb_ramp_unsure + n_given_obstacle_unsure + n_given_surface_problem_unsure, 0) AS n_given_unsure,
    COALESCE(n_given_curb_ramp_agree + n_given_curb_ramp_disagree + n_given_curb_ramp_unsure, 0) AS n_given_curb_ramp,
    COALESCE(n_given_curb_ramp_agree, 0) AS n_given_curb_ramp_agree,
    COALESCE(n_given_curb_ramp_disagree, 0) AS n_given_curb_ramp_disagree,
    COALESCE(n_given_curb_ramp_unsure, 0) AS n_given_curb_ramp_unsure,
    COALESCE(n_given_missing_curb_ramp_agree + n_given_missing_curb_ramp_disagree + n_given_missing_curb_ramp_unsure, 0) AS n_given_missing_curb_ramp,
    COALESCE(n_given_missing_curb_ramp_agree, 0) AS n_given_missing_curb_ramp_agree,
    COALESCE(n_given_missing_curb_ramp_disagree, 0) AS n_given_missing_curb_ramp_disagree,
    COALESCE(n_given_missing_curb_ramp_unsure, 0) AS n_given_missing_curb_ramp_unsure,
    COALESCE(n_given_obstacle_agree + n_given_obstacle_disagree + n_given_obstacle_unsure, 0) AS n_given_obstacle,
    COALESCE(n_given_obstacle_agree, 0) AS n_given_obstacle_agree,
    COALESCE(n_given_obstacle_disagree, 0) AS n_given_obstacle_disagree,
    COALESCE(n_given_obstacle_unsure, 0) AS n_given_obstacle_unsure,
    COALESCE(n_given_surface_problem_agree + n_given_surface_problem_disagree + n_given_surface_problem_unsure, 0) AS n_given_surface_problem,
    COALESCE(n_given_surface_problem_agree, 0) AS n_given_surface_problem_agree,
    COALESCE(n_given_surface_problem_disagree, 0) AS n_given_surface_problem_disagree,
    COALESCE(n_given_surface_problem_unsure, 0) AS n_given_surface_problem_unsure,
    COALESCE(audit_mission_counts.n_audit_mission, 0) AS n_audit_mission,
    COALESCE(validation_mission_counts.n_validation_mission, 0) AS n_validation_mission,
    user_stat.high_quality_manual,
    audited_distance.meters_audited,
    n_label,
    n_label_with_tag,
    n_label_with_description,
    n_label_with_severity,
    label_severity_min,
    label_severity_max,
    label_severity_mean,
    label_severity_sd,
    n_curb_ramp,
    n_curb_ramp_with_tag,
    n_curb_ramp_with_description,
    n_curb_ramp_with_severity,
    curb_ramp_severity_min,
    curb_ramp_severity_max,
    curb_ramp_severity_mean,
    curb_ramp_severity_sd,
    n_missing_curb_ramp,
    n_missing_curb_ramp_with_tag,
    n_missing_curb_ramp_with_description,
    n_missing_curb_ramp_with_severity,
    missing_curb_ramp_severity_min,
    missing_curb_ramp_severity_max,
    missing_curb_ramp_severity_mean,
    missing_curb_ramp_severity_sd,
    n_obstacle,
    n_obstacle_with_tag,
    n_obstacle_with_description,
    n_obstacle_with_severity,
    obstacle_severity_min,
    obstacle_severity_max,
    obstacle_severity_mean,
    obstacle_severity_sd,
    n_surface_problem,
    n_surface_problem_with_tag,
    n_surface_problem_with_description,
    n_surface_problem_with_severity,
    surface_problem_severity_min,
    surface_problem_severity_max,
    surface_problem_severity_mean,
    surface_problem_severity_sd,
    n_no_sidewalk,
    n_no_sidewalk_with_tag,
    n_no_sidewalk_with_description,
    n_no_sidewalk_with_severity,
    no_sidewalk_severity_min,
    no_sidewalk_severity_max,
    no_sidewalk_severity_mean,
    no_sidewalk_severity_sd,
    tutorial_times.tutorial_minutes,
    tutorial_error_count
-- SELECT COUNT(*)
-- The list of users.
FROM (
    SELECT user_id
    FROM (
        SELECT mission.user_id, COUNT(DISTINCT(label.label_id)) AS label_count, SUM(distance_progress) AS total_dist
        FROM mission
        INNER JOIN label ON mission.mission_id = label.mission_id
        WHERE label.deleted = FALSE
            AND label.tutorial = FALSE
            AND label.label_type_id < 5 -- 4 main label types only
        GROUP BY mission.user_id
    ) val_counts
    WHERE label_count > 9 AND total_dist > 0
) users
INNER JOIN user_role ON users.user_id = user_role.user_id
INNER JOIN role ON user_role.role_id = role.role_id
INNER JOIN user_stat ON users.user_id = user_stat.user_id
-- Validations given
LEFT JOIN (
    SELECT mission.user_id,
        COUNT(label_validation_id) AS n_validation_given,
        COUNT(CASE WHEN label_type_id = 1 AND validation_result = 1 THEN 1 END) AS n_given_curb_ramp_agree,
        COUNT(CASE WHEN label_type_id = 1 AND validation_result = 2 THEN 1 END) AS n_given_curb_ramp_disagree,
        COUNT(CASE WHEN label_type_id = 1 AND validation_result = 3 THEN 1 END) AS n_given_curb_ramp_unsure,
        COUNT(CASE WHEN label_type_id = 2 AND validation_result = 1 THEN 1 END) AS n_given_missing_curb_ramp_agree,
        COUNT(CASE WHEN label_type_id = 2 AND validation_result = 2 THEN 1 END) AS n_given_missing_curb_ramp_disagree,
        COUNT(CASE WHEN label_type_id = 2 AND validation_result = 3 THEN 1 END) AS n_given_missing_curb_ramp_unsure,
        COUNT(CASE WHEN label_type_id = 3 AND validation_result = 1 THEN 1 END) AS n_given_obstacle_agree,
        COUNT(CASE WHEN label_type_id = 3 AND validation_result = 2 THEN 1 END) AS n_given_obstacle_disagree,
        COUNT(CASE WHEN label_type_id = 3 AND validation_result = 3 THEN 1 END) AS n_given_obstacle_unsure,
        COUNT(CASE WHEN label_type_id = 4 AND validation_result = 1 THEN 1 END) AS n_given_surface_problem_agree,
        COUNT(CASE WHEN label_type_id = 4 AND validation_result = 2 THEN 1 END) AS n_given_surface_problem_disagree,
        COUNT(CASE WHEN label_type_id = 4 AND validation_result = 3 THEN 1 END) AS n_given_surface_problem_unsure
    FROM mission
    INNER JOIN label_validation ON mission.mission_id = label_validation.mission_id
    WHERE mission_type_id = 4 -- excludes validations through LabelMap
        AND label_type_id < 5 -- only consider 4 main label types
    GROUP BY mission.user_id
) validations_given ON users.user_id = validations_given.user_id
-- Audit mission counts.
LEFT JOIN (
    SELECT user_id, COUNT(mission_id) AS n_audit_mission
    FROM mission
    WHERE completed = TRUE
        AND mission_type_id = 2
    GROUP BY user_id
) audit_mission_counts ON users.user_id = audit_mission_counts.user_id
-- Validation mission counts.
LEFT JOIN (
    SELECT user_id, COUNT(mission_id) AS n_validation_mission
    FROM mission
    WHERE completed = TRUE
        AND mission_type_id = 4
    GROUP BY user_id
) validation_mission_counts ON users.user_id = validation_mission_counts.user_id
-- Meters audited.
INNER JOIN (
    SELECT user_id, SUM(distance_progress) AS meters_audited
    FROM mission
    WHERE mission_type_id = 2
    GROUP BY user_id
) audited_distance ON users.user_id = audited_distance.user_id
-- Label counts w/ and w/out severity, tags, and descriptions. Plus received validation counts.
INNER JOIN (
    SELECT user_id,
        COUNT(label_id) AS n_label,
        COUNT(CASE WHEN valid_tags > 0 THEN 1 END) AS n_label_with_tag,
        COUNT(CASE WHEN label_description_id IS NOT NULL THEN 1 END) AS n_label_with_description,
        COUNT(CASE WHEN label_severity_id IS NOT NULL THEN 1 END) AS n_label_with_severity,
        min(CASE WHEN label_severity_id IS NOT NULL THEN severity END) AS label_severity_min,
        max(CASE WHEN label_severity_id IS NOT NULL THEN severity END) AS label_severity_max,
        avg(CASE WHEN label_severity_id IS NOT NULL THEN severity END) AS label_severity_mean,
        stddev(CASE WHEN label_severity_id IS NOT NULL THEN severity END) AS label_severity_sd,
        COUNT(CASE WHEN label_type_id = 1 THEN 1 END) AS n_curb_ramp,
        COUNT(CASE WHEN label_type_id = 1 AND valid_tags > 0 THEN 1 END) AS n_curb_ramp_with_tag,
        COUNT(CASE WHEN label_type_id = 1 AND label_description_id IS NOT NULL THEN 1 END) AS n_curb_ramp_with_description,
        COUNT(CASE WHEN label_type_id = 1 AND label_severity_id IS NOT NULL THEN 1 END) AS n_curb_ramp_with_severity,
        min(CASE WHEN label_type_id = 1 AND label_severity_id IS NOT NULL THEN severity END) AS curb_ramp_severity_min,
        max(CASE WHEN label_type_id = 1 AND label_severity_id IS NOT NULL THEN severity END) AS curb_ramp_severity_max,
        avg(CASE WHEN label_type_id = 1 AND label_severity_id IS NOT NULL THEN severity END) AS curb_ramp_severity_mean,
        stddev(CASE WHEN label_type_id = 1 AND label_severity_id IS NOT NULL THEN severity END) AS curb_ramp_severity_sd,
        COUNT(CASE WHEN label_type_id = 2 THEN 1 END) AS n_missing_curb_ramp,
        COUNT(CASE WHEN label_type_id = 2 AND valid_tags > 0 THEN 1 END) AS n_missing_curb_ramp_with_tag,
        COUNT(CASE WHEN label_type_id = 2 AND label_description_id IS NOT NULL THEN 1 END) AS n_missing_curb_ramp_with_description,
        COUNT(CASE WHEN label_type_id = 2 AND label_severity_id IS NOT NULL THEN 1 END) AS n_missing_curb_ramp_with_severity,
        min(CASE WHEN label_type_id = 2 AND label_severity_id IS NOT NULL THEN severity END) AS missing_curb_ramp_severity_min,
        max(CASE WHEN label_type_id = 2 AND label_severity_id IS NOT NULL THEN severity END) AS missing_curb_ramp_severity_max,
        avg(CASE WHEN label_type_id = 2 AND label_severity_id IS NOT NULL THEN severity END) AS missing_curb_ramp_severity_mean,
        stddev(CASE WHEN label_type_id = 2 AND label_severity_id IS NOT NULL THEN severity END) AS missing_curb_ramp_severity_sd,
        COUNT(CASE WHEN label_type_id = 3 THEN 1 END) AS n_obstacle,
        COUNT(CASE WHEN label_type_id = 3 AND valid_tags > 0 THEN 1 END) AS n_obstacle_with_tag,
        COUNT(CASE WHEN label_type_id = 3 AND label_description_id IS NOT NULL THEN 1 END) AS n_obstacle_with_description,
        COUNT(CASE WHEN label_type_id = 3 AND label_severity_id IS NOT NULL THEN 1 END) AS n_obstacle_with_severity,
        min(CASE WHEN label_type_id = 3 AND label_severity_id IS NOT NULL THEN severity END) AS obstacle_severity_min,
        max(CASE WHEN label_type_id = 3 AND label_severity_id IS NOT NULL THEN severity END) AS obstacle_severity_max,
        avg(CASE WHEN label_type_id = 3 AND label_severity_id IS NOT NULL THEN severity END) AS obstacle_severity_mean,
        stddev(CASE WHEN label_type_id = 3 AND label_severity_id IS NOT NULL THEN severity END) AS obstacle_severity_sd,
        COUNT(CASE WHEN label_type_id = 4 THEN 1 END) AS n_surface_problem,
        COUNT(CASE WHEN label_type_id = 4 AND valid_tags > 0 THEN 1 END) AS n_surface_problem_with_tag,
        COUNT(CASE WHEN label_type_id = 4 AND label_description_id IS NOT NULL THEN 1 END) AS n_surface_problem_with_description,
        COUNT(CASE WHEN label_type_id = 4 AND label_severity_id IS NOT NULL THEN 1 END) AS n_surface_problem_with_severity,
        min(CASE WHEN label_type_id = 4 AND label_severity_id IS NOT NULL THEN severity END) AS surface_problem_severity_min,
        max(CASE WHEN label_type_id = 4 AND label_severity_id IS NOT NULL THEN severity END) AS surface_problem_severity_max,
        avg(CASE WHEN label_type_id = 4 AND label_severity_id IS NOT NULL THEN severity END) AS surface_problem_severity_mean,
        stddev(CASE WHEN label_type_id = 4 AND label_severity_id IS NOT NULL THEN severity END) AS surface_problem_severity_sd,
        COUNT(CASE WHEN label_type_id = 7 THEN 1 END) AS n_no_sidewalk,
        COUNT(CASE WHEN label_type_id = 7 AND valid_tags > 0 THEN 1 END) AS n_no_sidewalk_with_tag,
        COUNT(CASE WHEN label_type_id = 7 AND label_description_id IS NOT NULL THEN 1 END) AS n_no_sidewalk_with_description,
        COUNT(CASE WHEN label_type_id = 7 AND label_severity_id IS NOT NULL THEN 1 END) AS n_no_sidewalk_with_severity,
        min(CASE WHEN label_type_id = 7 AND label_severity_id IS NOT NULL THEN severity END) AS no_sidewalk_severity_min,
        max(CASE WHEN label_type_id = 7 AND label_severity_id IS NOT NULL THEN severity END) AS no_sidewalk_severity_max,
        avg(CASE WHEN label_type_id = 7 AND label_severity_id IS NOT NULL THEN severity END) AS no_sidewalk_severity_mean,
        stddev(CASE WHEN label_type_id = 7 AND label_severity_id IS NOT NULL THEN severity END) AS no_sidewalk_severity_sd,
        COUNT(CASE WHEN label_type_id < 5 AND agree_count + disagree_count + notsure_count > 0 THEN 1 END) AS n_validation_received, -- only include 4 main label types
        COUNT(CASE WHEN validation_result = 1 AND label_type_id = 1 THEN 1 END) AS n_received_curb_ramp_agree,
        COUNT(CASE WHEN validation_result = 2 AND label_type_id = 1 THEN 1 END) AS n_received_curb_ramp_disagree,
        COUNT(CASE WHEN validation_result = 3 AND label_type_id = 1 THEN 1 END) AS n_received_curb_ramp_unsure,
        COUNT(CASE WHEN validation_result = 4 AND label_type_id = 1 THEN 1 END) AS n_received_curb_ramp_tie,
        COUNT(CASE WHEN validation_result = 1 AND label_type_id = 2 THEN 1 END) AS n_received_missing_curb_ramp_agree,
        COUNT(CASE WHEN validation_result = 2 AND label_type_id = 2 THEN 1 END) AS n_received_missing_curb_ramp_disagree,
        COUNT(CASE WHEN validation_result = 3 AND label_type_id = 2 THEN 1 END) AS n_received_missing_curb_ramp_unsure,
        COUNT(CASE WHEN validation_result = 4 AND label_type_id = 2 THEN 1 END) AS n_received_missing_curb_ramp_tie,
        COUNT(CASE WHEN validation_result = 1 AND label_type_id = 3 THEN 1 END) AS n_received_obstacle_agree,
        COUNT(CASE WHEN validation_result = 2 AND label_type_id = 3 THEN 1 END) AS n_received_obstacle_disagree,
        COUNT(CASE WHEN validation_result = 3 AND label_type_id = 3 THEN 1 END) AS n_received_obstacle_unsure,
        COUNT(CASE WHEN validation_result = 4 AND label_type_id = 3 THEN 1 END) AS n_received_obstacle_tie,
        COUNT(CASE WHEN validation_result = 1 AND label_type_id = 4 THEN 1 END) AS n_received_surface_problem_agree,
        COUNT(CASE WHEN validation_result = 2 AND label_type_id = 4 THEN 1 END) AS n_received_surface_problem_disagree,
        COUNT(CASE WHEN validation_result = 3 AND label_type_id = 4 THEN 1 END) AS n_received_surface_problem_unsure,
        COUNT(CASE WHEN validation_result = 4 AND label_type_id = 4 THEN 1 END) AS n_received_surface_problem_tie
    FROM (
        SELECT user_id,
            label.label_id,
            label.label_type_id,
            label_severity.label_severity_id,
            severity,
            COUNT(label_tag.label_tag_id) AS valid_tags,
            label_description_id,
            agree_count,
            disagree_count,
            notsure_count,
            CASE WHEN agree_count > disagree_count AND agree_count > notsure_count THEN 1
                WHEN disagree_count > agree_count AND disagree_count > notsure_count THEN 2
                WHEN notsure_count > agree_count AND notsure_count > disagree_count THEN 3
                ELSE 4 END AS validation_result
        FROM mission
        INNER JOIN label ON mission.mission_id = label.mission_id
        LEFT JOIN label_severity ON label.label_id = label_severity.label_id
        LEFT JOIN label_tag ON label.label_id = label_tag.label_id
        LEFT JOIN label_description ON label.label_id = label_description.label_id
        WHERE tutorial = FALSE
            AND deleted = FALSE
            AND mission_type_id = 2
        GROUP BY user_id, label.label_id, label.label_type_id, label_severity_id, severity, label_description_id
    ) labs
    GROUP BY user_id
) label_severity_counts ON users.user_id = label_severity_counts.user_id
-- Tutorial completion time.
LEFT JOIN (
    SELECT user_audit_times.user_id,
        CAST(extract( second from SUM(diff) ) / 60 +
            extract( minute from SUM(diff) ) +
            extract( hour from SUM(diff) ) * 60 AS decimal(10,2)) AS tutorial_minutes
    FROM
    (
        SELECT tutorial_task.user_id,
            (timestamp - LAG(timestamp, 1) OVER(PARTITION BY user_id ORDER BY timestamp)) AS diff
        FROM
        (
            SELECT DISTINCT ON (audit_task.user_id)
                audit_task.audit_task_id, audit_task.user_id
            FROM audit_task_interaction
            INNER JOIN audit_task ON audit_task_interaction.audit_task_id = audit_task.audit_task_id
            INNER JOIN user_role ON audit_task.user_id = user_role.user_id
            INNER JOIN role ON user_role.role_id = role.role_id
            LEFT JOIN (
                SELECT audit_task_id
                FROM audit_task_interaction
                WHERE action = 'Onboarding_Skip'
            ) skipped_tutorials ON audit_task.audit_task_id = skipped_tutorials.audit_task_id
            WHERE action = 'Onboarding_End'
                AND skipped_tutorials.audit_task_id IS NULL
                AND audit_task.user_id <> '97760883-8ef0-4309-9a5e-0c086ef27573'
            ORDER BY audit_task.user_id, audit_task_interaction.timestamp, audit_task_interaction.audit_task_interaction_id
        ) tutorial_task
        INNER JOIN audit_task_interaction ON tutorial_task.audit_task_id = audit_task_interaction.audit_task_id
        WHERE action IN ('ViewControl_MouseDown', 'LabelingCanvas_MouseDown', 'Onboarding_Transition')
    ) user_audit_times
    WHERE diff < '00:05:00.000' AND diff > '00:00:00.000'
    GROUP BY user_audit_times.user_id
) tutorial_times ON users.user_id = tutorial_times.user_id
-- Tutorial error counts.
LEFT JOIN (
    SELECT user_id,
        COUNT(CASE WHEN note LIKE '%redo%' OR (note LIKE 'onboardingTransition:select-label-type-%' AND count > 2) THEN 1 END) AS tutorial_error_count
    FROM (
        SELECT user_id, action, note, COUNT(*)
        FROM (
            SELECT DISTINCT ON (audit_task.user_id)
                audit_task.audit_task_id, audit_task.user_id
            FROM audit_task_interaction
            INNER JOIN audit_task ON audit_task_interaction.audit_task_id = audit_task.audit_task_id
            INNER JOIN user_role ON audit_task.user_id = user_role.user_id
            INNER JOIN role ON user_role.role_id = role.role_id
            LEFT JOIN (
                SELECT audit_task_id
                FROM audit_task_interaction
                WHERE action = 'Onboarding_Skip'
            ) skipped_tutorials ON audit_task.audit_task_id = skipped_tutorials.audit_task_id
            WHERE action = 'Onboarding_End'
                AND skipped_tutorials.audit_task_id IS NULL
                AND audit_task.user_id <> '97760883-8ef0-4309-9a5e-0c086ef27573'
            ORDER BY audit_task.user_id, audit_task_interaction.timestamp, audit_task_interaction.audit_task_interaction_id
        ) tutorial_task
        INNER JOIN audit_task_interaction ON tutorial_task.audit_task_id = audit_task_interaction.audit_task_id
        WHERE action = 'Onboarding_Transition'
        GROUP BY user_id, action, note
    ) action_counts
    GROUP BY user_id
) tutorial_error_counts ON users.user_id = tutorial_error_counts.user_id
ORDER BY users.user_id;


-- INTERACTIONS QUERY -- output as ml-<city-name>-interaction-stats.csv
SELECT users.user_id,
    COALESCE(n_pano_visited, 0) AS n_pano_visited,
    COALESCE(n_pano_with_label, 0) AS n_pano_with_label,
    COALESCE(n_keyboard_interaction, 0) AS n_keyboard_interaction,
    COALESCE(n_pan_interaction, 0) AS n_pan_interaction
FROM (
    SELECT user_id
    FROM (
        SELECT mission.user_id, COUNT(DISTINCT(label.label_id)) AS label_count, SUM(distance_progress) AS total_dist
        FROM mission
        INNER JOIN label ON mission.mission_id = label.mission_id
        WHERE label.deleted = FALSE
            AND label.tutorial = FALSE
            AND label.label_type_id < 5 -- 4 main label types only
        GROUP BY mission.user_id
    ) val_counts
    WHERE label_count > 9 AND total_dist > 0
) users
LEFT JOIN (
    SELECT user_id,
        SUM(CASE WHEN action = 'LowLevelEvent_keyup' THEN count END) AS n_keyboard_interaction,
        SUM(CASE WHEN action = 'POV_Changed' THEN count END) AS n_pan_interaction
    FROM (
        SELECT actions.user_id, actions.action, COALESCE(count, 0) AS count
        FROM (
            SELECT action, user_id
            FROM audit_task_interaction
            INNER JOIN audit_task ON audit_task_interaction.audit_task_id = audit_task.audit_task_id
            WHERE action IN ('LowLevelEvent_keyup', 'POV_Changed')
            GROUP BY action, user_id
        ) actions
        LEFT JOIN (
            SELECT user_id, action, COUNT(*) AS count
            FROM audit_task
            INNER JOIN audit_task_interaction ON audit_task.audit_task_id = audit_task_interaction.audit_task_id
            WHERE action IN ('LowLevelEvent_keyup', 'POV_Changed')
                AND gsv_panorama_id NOT IN ('tutorial', 'afterWalkTutorial', 'stxXyCKAbd73DmkM2vsIHA', 'PTHUzZqpLdS1nTixJMoDSw')
            GROUP BY user_id, action
        ) action_counts ON actions.user_id = action_counts.user_id AND actions.action = action_counts.action
    ) interaction_counts
    GROUP BY user_id
) interaction_stats ON users.user_id = interaction_stats.user_id
LEFT JOIN (
    SELECT user_id,
        COUNT(DISTINCT(gsv_panorama_id)) AS n_pano_visited
    FROM audit_task_interaction
    INNER JOIN audit_task ON audit_task_interaction.audit_task_id = audit_task.audit_task_id
    WHERE action = 'PanoId_Changed'
    AND gsv_panorama_id NOT IN ('tutorial', 'afterWalkTutorial', 'stxXyCKAbd73DmkM2vsIHA', 'PTHUzZqpLdS1nTixJMoDSw')
    GROUP BY user_id
) pano_counts ON users.user_id = pano_counts.user_id
LEFT JOIN (
    SELECT user_id, COUNT(DISTINCT(gsv_panorama_id)) AS n_pano_with_label
    FROM label
    INNER JOIN mission
    ON label.mission_id = mission.mission_id
    WHERE deleted = FALSE
        AND tutorial = FALSE
        AND gsv_panorama_id NOT IN ('tutorial', 'afterWalkTutorial', 'stxXyCKAbd73DmkM2vsIHA', 'PTHUzZqpLdS1nTixJMoDSw')
    GROUP BY user_id
) pano_with_label_counts ON users.user_id = pano_with_label_counts.user_id;

-- ADD ON TO INTERACTIONS QUERY TO GET PANOS VISITED, TOO SLOW FOR SEATTLE (and probably SPGG by now)
-- 43 minutes to add index on local dump (40 million records)
-- LEFT JOIN (
--     SELECT user_id, COUNT(DISTINCT(gsv_panorama_id)) AS panos_visited
--     FROM audit_task
--     INNER JOIN audit_task_interaction ON audit_task.audit_task_id = audit_task_interaction.audit_task_id
--     GROUP BY user_id
-- ) pano_counts ON users.user_id = pano_counts.user_id



-- INTERACTIONS QUERY THAT TAKES TOO LONG TO RUN
SELECT users.user_id,
    panos_visited,
    --mousemove_per_pano_min,
    --mousemove_per_pano_max,
    --mousemove_per_pano_avg,
    --mousemove_per_pano_sd,
    keystroke_per_pano_min,
    keystroke_per_pano_max,
    keystroke_per_pano_avg,
    keystroke_per_pano_sd
    --pans_per_pano_min,
    --pans_per_pano_max,
    --pans_per_pano_avg,
    --pans_per_pano_sd
FROM (
    SELECT user_id
    FROM (
        SELECT mission.user_id, COUNT(DISTINCT(label.label_id)) AS label_count, SUM(distance_progress) AS total_dist
        FROM mission
        INNER JOIN label ON mission.mission_id = label.mission_id
        WHERE label.deleted = FALSE
            AND label.tutorial = FALSE
            AND label.label_type_id < 5 -- 4 main label types only
        GROUP BY mission.user_id
    ) val_counts
    WHERE label_count > 9 AND total_dist > 0
) users
INNER JOIN (
    SELECT user_id,
        COUNT(*) AS panos_visited, -- divide by n for number of interactions, or count earlier
        -- min(CASE WHEN action = 'LowLevelEvent_mousemove' THEN count END) AS mousemove_per_pano_min,
        -- max(CASE WHEN action = 'LowLevelEvent_mousemove' THEN count END) AS mousemove_per_pano_max,
        -- avg(CASE WHEN action = 'LowLevelEvent_mousemove' THEN count END) AS mousemove_per_pano_avg,
        -- stddev(CASE WHEN action = 'LowLevelEvent_mousemove' THEN count END) AS mousemove_per_pano_sd,
        min(CASE WHEN action = 'LowLevelEvent_keyup' THEN count END) AS keystroke_per_pano_min,
        max(CASE WHEN action = 'LowLevelEvent_keyup' THEN count END) AS keystroke_per_pano_max,
        avg(CASE WHEN action = 'LowLevelEvent_keyup' THEN count END) AS keystroke_per_pano_avg,
        stddev(CASE WHEN action = 'LowLevelEvent_keyup' THEN count END) AS keystroke_per_pano_sd
        -- min(CASE WHEN action = 'POV_Changed' THEN count END) AS pans_per_pano_min,
        -- max(CASE WHEN action = 'POV_Changed' THEN count END) AS pans_per_pano_max,
        -- avg(CASE WHEN action = 'POV_Changed' THEN count END) AS pans_per_pano_avg,
        -- stddev(CASE WHEN action = 'POV_Changed' THEN count END) AS pans_per_pano_sd
    FROM (
        SELECT actions.user_id, panos.gsv_panorama_id, actions.action, COUNT(all_actions.gsv_panorama_id)
        FROM (
            SELECT action, user_id
            FROM audit_task_interaction
            INNER JOIN audit_task ON audit_task_interaction.audit_task_id = audit_task.audit_task_id
            WHERE action IN ('LowLevelEvent_keyup')
            GROUP BY action, user_id
        ) actions
        INNER JOIN (
            SELECT gsv_panorama_id, user_id
            FROM audit_task_interaction
            INNER JOIN audit_task ON audit_task_interaction.audit_task_id = audit_task.audit_task_id
            WHERE gsv_panorama_id IS NOT NULL
                AND gsv_panorama_id NOT IN ('tutorial', 'afterWalkTutorial', 'stxXyCKAbd73DmkM2vsIHA', 'PTHUzZqpLdS1nTixJMoDSw')
            GROUP BY gsv_panorama_id, user_id
        ) panos ON actions.user_id = panos.user_id
        LEFT JOIN (
            SELECT user_id, gsv_panorama_id, action
            FROM audit_task
            INNER JOIN audit_task_interaction ON audit_task.audit_task_id = audit_task_interaction.audit_task_id
            WHERE action IN ('LowLevelEvent_keyup')
                AND gsv_panorama_id NOT IN ('tutorial', 'afterWalkTutorial', 'stxXyCKAbd73DmkM2vsIHA', 'PTHUzZqpLdS1nTixJMoDSw')
        ) all_actions ON actions.user_id = all_actions.user_id AND actions.action = all_actions.action AND panos.gsv_panorama_id = all_actions.gsv_panorama_id
        GROUP BY actions.user_id, actions.action, panos.gsv_panorama_id
    ) interaction_counts
    GROUP BY user_id
) interaction_stats ON users.user_id = interaction_stats.user_id


-- USERS QUERY -- output as ml-<city-name>-users.csv
SELECT users_with_stats.user_id, username, email
FROM (
    SELECT mission.user_id, COUNT(DISTINCT(label.label_id)) AS label_count, SUM(distance_progress) AS total_dist
    FROM mission
    INNER JOIN label ON mission.mission_id = label.mission_id
    WHERE label.deleted = FALSE
        AND label.tutorial = FALSE
        AND label.label_type_id < 5 -- 4 main label types only
    GROUP BY mission.user_id
) users_with_stats
INNER JOIN sidewalk_user ON users_with_stats.user_id = sidewalk_user.user_id
WHERE label_count > 9 AND total_dist > 0
ORDER BY users_with_stats.user_id;
