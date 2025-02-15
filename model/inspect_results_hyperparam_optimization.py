import sqlite3


def show_all_database_objects(db_file):
    """ Inspects an Optuna hyperparameter optimization results database (SQLite)
    and displays information about all objects (tables, views, indexes, triggers)
    stored within it.

    Args:
        db_file: The path to the SQLite database file.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Get a list of all objects with a non-null type
    cursor.execute("SELECT name, type FROM sqlite_master WHERE type IS NOT NULL;")
    objects = cursor.fetchall()

    for name, type in objects:
        print(f"\nObject: {name}")
        print(f"Type: {type}")

        if type in ('table', 'view'):
            cursor.execute(f"SELECT * FROM {name}")
            rows = cursor.fetchall()

            # Print column headers
            column_names = [description[0] for description in cursor.description]
            print('\t'.join(column_names))

            # Print rows with formatted output
            for row in rows:
                print('\t'.join(str(value) for value in row))
                
        elif type == 'index':
            # Print index details
            cursor.execute(f"PRAGMA index_info({name})")
            index_info = cursor.fetchall()
            print("Index Information:")
            for info in index_info:
                print(info)
                
        elif type == 'trigger':
            # Print trigger definition
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='trigger' AND name='{name}'")
            trigger_definition = cursor.fetchone()[0]
            print("Trigger Definition:")
            print(trigger_definition)
            
        else:
            print("Not a table, view, or index. Cannot display contents.")

    conn.close()
    
    
    
def find_best_trial_and_params(db_file):
    """Finds the best trial number and its corresponding hyperparameters from an
    Optuna hyperparameter optimization trial stored in a SQLite database.

    The "best" trial is determined by the most *negative* `value_json`
    (since Optuna minimizes the objective function, which is negative log
    likelihood, by default).

    Args:
        db_file: The path to the Optuna SQLite database file.

    Returns:
        tuple: A tuple containing the best trial ID (int) and a dictionary
               of the best hyperparameters.  Returns (None, None) if no best
               trial is found.

    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Find the trial with the most negative value_json
    cursor.execute("SELECT trial_id, value_json FROM trial_user_attributes ORDER BY value_json DESC LIMIT 1")
    best_trial_id, value_json = cursor.fetchone()

    # Retrieve the parameters for the best trial
    cursor.execute("""
        SELECT param_name, param_value
        FROM trial_params
        WHERE trial_id = ?
    """, (best_trial_id,))

    best_params = {}
    for param_name, param_value in cursor.fetchall():
        best_params[param_name] = param_value
        print(param_name)

    print("The best trial was:")
    print(f"Trial ID: {best_trial_id}")
    print(f"Value objective: {value_json}")
    print(f"Best Params: {best_params}")
        
    conn.close()