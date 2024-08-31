import csv
import math
import os
from tempfile import NamedTemporaryFile

import numpy as np
import psycopg2 as pg
from psycopg2 import Error
from psycopg2 import sql
import duckdb
from parameters import Parameters


class Database:
    schemas = [
        """
		CREATE TABLE IF NOT EXISTS db.public.training (
			run_id UUID,
			generation INT,
			team_id UUID,
			is_finished BOOLEAN,
			reward FLOAT8,
			time_step INT,
			time FLOAT8,
			action INT,
			PRIMARY KEY (run_id, generation, team_id, time_step)
		)
		""",
        """
			CREATE TABLE IF NOT EXISTS db.public.teams (
                run_id UUID,
                id UUID,
                lucky_breaks INT,
                PRIMARY KEY (run_id, id)
			);
		""",
        """
			CREATE TABLE IF NOT EXISTS db.public.programs (
                run_id UUID,
				id UUID, 
				team_id UUID,
				action VARCHAR,
				pointer UUID,
				CONSTRAINT action_or_pointer_check CHECK (
						(action IS NOT NULL AND pointer IS NULL) OR
						(action IS NULL AND pointer IS NOT NULL)
				),
				PRIMARY KEY (run_id, id, team_id)
			);
		""",
        """
            CREATE TABLE IF NOT EXISTS db.public.cpu_utilization (
                run_id UUID,
                time FLOAT8,
                worker VARCHAR,
                core INT,
                utilization FLOAT8,
                PRIMARY KEY (run_id, time, worker, core)
            );
        """,
        """
			CREATE TABLE IF NOT EXISTS db.public.time_monitor (
				run_id UUID,
				generation INT,
				time FLOAT8,
				PRIMARY KEY (run_id, generation)
			);
		""",
        """
            CREATE TABLE IF NOT EXISTS db.public.observations (
                run_id UUID,
                time FLOAT8,
                observation FLOAT8[]
            );
        """,
        """
            CREATE TABLE IF NOT EXISTS db.public.diversity_cache (
                run_id UUID,
                time FLOAT8,
                team_id UUID,
                profile INT[]
            );
        """,
        """
            CREATE TABLE IF NOT EXISTS db.public.compute_configs (
                run_id UUID PRIMARY KEY,
                team_distribution VARCHAR,
                batch_sizes VARCHAR
            );
        """
    ]

    @classmethod
    def connect(cls, user, password, host, port, database):
        try:
            duckdb.sql("INSTALL postgres;")
            duckdb.sql("LOAD postgres;")
            duckdb.sql(f"ATTACH 'dbname={database} user={user} host={host} password={password}' AS db (TYPE POSTGRES);")

            for schema in cls.schemas:
                duckdb.sql(schema)

        except (Exception, Error) as error:
            print("Error while connecting to database", error)

    @classmethod
    def disconnect(cls):
        duckdb.sql("DETACH db;")

    @classmethod
    def clear(cls):
        duckdb.sql("""
		DROP TABLE IF EXISTS db.public.instructions;
		DROP TABLE IF EXISTS db.public.programs;
		DROP TABLE IF EXISTS db.public.teams;
		DROP TABLE IF EXISTS db.public.training;
		DROP TABLE IF EXISTS db.public.cpu_utilization;
		DROP TABLE IF EXISTS db.public.observations;
		DROP TABLE IF EXISTS db.public.time_monitor;
		DROP TABLE IF EXISTS db.public.diversity_cache;
		DROP TABLE IF EXISTS db.public.compute_configs;
		""")

    @staticmethod
    def add_time_monitor_data(run_id, generation, time):
        duckdb.sql(
            f"INSERT INTO db.public.time_monitor (run_id, generation, time) VALUES ('{run_id}', {generation}, {time});")

    @staticmethod
    def add_cpu_utilization_data(data):
        for row in data:
            run_id = row['run_id']
            time = row['time']
            worker = row['worker']
            core = row['core']
            utilization = row['utilization']
            query = f"INSERT INTO db.public.cpu_utilization (run_id, time, worker, core, utilization) VALUES ('{run_id}', {time}, '{worker}', {core}, {utilization});"
            duckdb.sql(query)

    @staticmethod
    def add_training_data(data):
        try:
            # Create a temporary CSV file
            with NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
                writer = csv.writer(temp_file)

                # Write data rows to the CSV file
                for row in data:
                    writer.writerow([
                        row['run_id'],
                        row['generation'],
                        row['team_id'],
                        row['is_finished'],
                        row['reward'],
                        row['time_step'],
                        row['time'],
                        row['action']
                    ])

                temp_file_path = temp_file.name

            # Use the COPY command to load data from the CSV file
            duckdb.query(f"""
                    COPY db.public.training FROM '{temp_file_path}' (FORMAT CSV);
                """)

            os.remove(temp_file_path)

        except Exception as e:
            # Handle any exceptions
            print(f"An error occurred: {e}")
            raise e

    @staticmethod
    def add_compute_config(run_id, team_distribution, batch_sizes):
        query = f"INSERT INTO db.public.compute_configs VALUES ('{run_id}', '{team_distribution}', '{batch_sizes}');"
        print(query)
        duckdb.sql(query)

    @staticmethod
    def add_team(run_id, team):
        duckdb.sql(f"INSERT INTO db.public.teams (run_id, id, lucky_breaks) VALUES ('{run_id}', '{team.id}', 0);")

    @staticmethod
    def remove_team(run_id, team):
        duckdb.sql(f"DELETE FROM db.public.teams WHERE id = '{team.id}' AND run_id = '{run_id}';")

    @staticmethod
    def remove_program(run_id, learner, team):
        duckdb.sql(
            f"DELETE FROM db.public.programs WHERE run_id = '{run_id}' AND id = '{learner.id}' AND team_id = '{team.id}';")

    @staticmethod
    def remove_programs(run_id, removed_programs_and_teams):
        duckdb.query("""
        CREATE TEMPORARY TABLE temp_delete_ids (
        program_id INT,
        team_id INT
        ); 
        """)

        values_clause = ', '.join(f"({program_id}, {team_id})" for program_id, team_id in removed_programs_and_teams)
        insert_statement = f"""
        INSERT INTO db.public.temp_delete_ids (program_id, team_id)
        VALUES {values_clause};"""

        duckdb.sql(f"""
        DELETE p
        FROM db.public.programs p
        JOIN db.public.temp_delete_ids t ON p.id = t.program_id AND p.team_id = t.team_id
        WHERE p.run_id = '{run_id}';
        """)

        duckdb.sql(f"""
        DROP TEMPORARY TABLE temp_delete_ids;
        """)

    @staticmethod
    def remove_teams(run_id, removed_teams):
        duckdb.query("""
                    CREATE TEMPORARY TABLE temp_remove_teams (
                        team_id UUID
                    );
                """)

        # Prepare the VALUES clause for batch insertion
        values_clause = ', '.join(f"('{team_id}')" for team_id in removed_teams)

        # Form the complete INSERT statement
        insert_statement = f"""
                INSERT INTO db.public.temp_remove_teams (team_id)
                VALUES {values_clause};
                """

        # Execute the INSERT statement
        duckdb.query(insert_statement)

        # Perform the batch DELETE operation
        delete_statement = f"""
                DELETE t
                FROM db.public.teams t
                JOIN db.public.temp_remove_teams r
                ON t.id = r.team_id
                WHERE t.run_id = '{run_id}';
                """

        # Execute the DELETE statement
        duckdb.query(delete_statement)

        # Drop the temporary table
        duckdb.query("DROP TABLE temp_remove_teams;")

    @staticmethod
    def add_program(run_id, program, team):
        duckdb.sql(f"""
            INSERT INTO db.public.programs (run_id, id, team_id, action, pointer)
            VALUES ('{run_id}', '{program.id}', '{team.id}', '{program.action}', NULL);""")

    @staticmethod
    def update_program(run_id, program, team, action, pointer):
        if not action:
            action = "NULL"
        else:
            action = f"'{action}'"

        if not pointer:
            pointer = "NULL"
        else:
            pointer = f"'{pointer}'"

        duckdb.sql(f"""
        UPDATE db.public.programs
        SET
            action = {action},
            pointer = {pointer}
        WHERE run_id = '{run_id}'
        AND id = '{program.id}'
        AND team_id = '{team.id}'""")

    @staticmethod
    def get_teams(run_id):
        return duckdb.sql(f"""SELECT * FROM db.public.teams WHERE run_id = '{run_id}'""")

    @staticmethod
    def get_root_teams(run_id):
        return duckdb.sql(f"""
            WITH programs_pointing_to_teams AS (
                SELECT pointer FROM db.public.programs 
                WHERE pointer IS NOT NULL
                AND run_id = '{run_id}'
            )

            SELECT * FROM db.public.teams
            WHERE id NOT IN (SELECT pointer FROM programs_pointing_to_teams)
            AND run_id = '{run_id}'
            """).df()['id'].tolist()

    @staticmethod
    def get_ranked_teams(run_id, generation):
        return duckdb.sql(f"""
                WITH team_cumulative_rewards AS (
                  SELECT generation,
                         team_id,
                         SUM(reward) AS cumulative_reward
                  FROM db.public.training
                  WHERE generation = '{generation}'
                  AND run_id = '{run_id}'
                  GROUP BY generation, team_id
				)

				SELECT generation,
			           team_id,
					   cumulative_reward,
					   ROW_NUMBER() OVER (PARTITION BY generation ORDER BY cumulative_reward DESC) AS rank
				FROM team_cumulative_rewards
				WHERE generation={generation}""").df()

    @staticmethod
    def update_team(run_id, team, lucky_breaks):
        return duckdb.sql(f"""
        UPDATE db.public.teams SET lucky_breaks = '{lucky_breaks}' 
        WHERE id = '{team.id}'
        AND run_id = '{run_id}';
        """)

    @staticmethod
    def add_observation(run_id, time, observation):
        observation = ', '.join(map(str, observation))

        query = f"""
        INSERT INTO db.public.observations (run_id, time, observation)
        VALUES ('{run_id}', {time}, ARRAY[{observation}])
       """

        return duckdb.sql(query)

    @staticmethod
    def add_profile(run_id, team, time, profile):
        profile = ', '.join(map(str, profile))

        query = f"""
        INSERT INTO db.public.diversity_cache (run_id, team_id, time, profile)
        VALUES ('{run_id}', '{team.id}', {time}, ARRAY[{profile}])
        """

        return duckdb.sql(query)

    @staticmethod
    def get_diversity_cache(run_id):
        return duckdb.query(f"""
        SELECT * FROM db.public.observations
        WHERE run_id = '{run_id}'
        ORDER BY time DESC
        LIMIT {Parameters.DIVERSITY_CACHE_SIZE} 
        """).df()['observation'].to_list()

    @staticmethod
    def get_diversity_profiles(run_id):
        profiles = []

        query_results = duckdb.query(f"""
            SELECT profile FROM db.public.diversity_cache
            WHERE run_id = '{run_id}'
            ORDER BY time DESC
            LIMIT {Parameters.DIVERSITY_CACHE_SIZE}
        """).df()['profile']

        for profile in query_results:
            if isinstance(profile, np.ndarray):
                profile = profile.tolist()
            profiles.append(profile)

        return profiles

    @staticmethod
    def get_survivor_ids(run_id, generation):
        survivor_count = math.floor(Parameters.POPGAP * Parameters.POPULATION_SIZE)
        sorted_team_ids = Database.get_ranked_teams(run_id, generation).sort_values('rank')['team_id']
        survivor_ids = sorted_team_ids[:survivor_count].to_list()
        return survivor_ids
