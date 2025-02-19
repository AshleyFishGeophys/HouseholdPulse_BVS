import sqlite3


def show_all_database_objects(db_file: str) -> None:
    """Inspects an Optuna hyperparameter optimization results database (SQLite)
    and displays information about all objects (tables, views, indexes, triggers)
    stored within it.

    Args:
        db_file (str):
            The path to the Optuna trial SQLite database file.

    Returns:
        None

    Typical Usage Example:
        During the Optuna hyperparameter optimization, it stores all of the trail info
        into a database with the trail name and date/time. Import that DB here in order
        to inspect its contents. 
        show_all_database_objects("study.db") 
        # Replace "study.db" with your database file path
    """

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Get a list of all objects (tables, views, indexes, triggers) with their types
        # Execute the SQL query to select name and type from sqlite_master where type is not null
        cursor.execute("SELECT name, type FROM sqlite_master WHERE type IS NOT NULL;")
        objects = cursor.fetchall() # Fetch all objects (name and type) as a list of tuples

        for name, type in objects: # Iterate through each database object
            print(f"\nObject: {name}") # Print the name of the object
            print(f"Type: {type}") # Print the type of the object

            # Check if the object is a table or a view
            if type in ('table', 'view'):
                # Select all data from the table or view
                cursor.execute(f"SELECT * FROM {name}")
                # Fetch all rows from the table or view 
                rows = cursor.fetchall()

                # Get column names from the cursor description
                column_names = [description[0] for description in cursor.description]
                print('\t'.join(column_names))

                # Print rows with formatted output
                # Iterate through each row of data
                for row in rows:
                    # Convert each value in the row to a string and join them with tabs
                    print('\t'.join(str(value) for value in row))

            # Check if the object is an index
            elif type == 'index':
                # Print index details
                cursor.execute(f"PRAGMA index_info({name})")
                # Fetch index information 
                index_info = cursor.fetchall()
                print("Index Information:")
                for info in index_info: # Iterate through index information
                    print(info)

            elif type == 'trigger': # Check if the object is a trigger
                # Get the SQL definition of the trigger
                # Select the SQL definition from sqlite_master for the trigger
                cursor.execute(
                    f"SELECT sql FROM sqlite_master WHERE type='trigger' AND name='{name}'"
                )
                 # Fetch the trigger definition as a string
                trigger_definition = cursor.fetchone()[0] 
                print("Trigger Definition:")
                print(trigger_definition)

            # If the object is not a table, view, index, or trigger
            else:
                print("Not a table, view, or index. Cannot display contents.")

    except sqlite3.Error as e:  # Handle potential SQLite errors
        print(f"An error occurred: {e}")  # Print the error message

    finally:
        if conn:  # Ensure the connection is closed even if an error occurs
            conn.close()  # Close the database connection    
    
    
def find_best_trial_and_params(db_file: str) -> None:
    """Finds the best trial number and its corresponding hyperparameters from an
    Optuna hyperparameter optimization trial stored in a SQLite database.

    The "best" trial is determined by the most *negative* `value_json`
    (since Optuna minimizes the objective function, which is negative log
    likelihood, by default).

    Args:
        db_file (str):
            The path to the Optuna trial SQLite database file.

    Returns:
        None 
        
    Typical Usage Example: 
        Find the best trial ID (int) and print the dictionary of the
        best hyperparameters. 

    """
    try: 
        conn = sqlite3.connect(db_file) # Establish a connection to the SQLite database
        cursor = conn.cursor() # Create a cursor object to execute SQL queries

        # Find the trial with the most negative value_json
        # Order by ASC to get the MIN value
        cursor.execute(
            "SELECT trial_id, value_json FROM trial_user_attributes ORDER BY value_json DESC LIMIT 1"
        ) 
        # Fetch the best trial ID and value_json.
        best_trial_id, value_json = cursor.fetchone()

        # Retrieve the parameters for the best trial
        cursor.execute("""
            SELECT param_name, param_value
            FROM trial_params
            WHERE trial_id = ?
        """, (best_trial_id,))

        # Initialize an empty dictionary to store the best hyperparameters
        best_params = {} 
        # Iterate through the fetched parameters
        for param_name, param_value in cursor.fetchall():
            # Store the parameter name and value in the dictionary
            best_params[param_name] = param_value
            print(param_name)

        print("The best trial was:")
        print(f"Trial ID: {best_trial_id}")
        print(f"Value objective: {value_json}")
        print(f"Best Params: {best_params}")

    except sqlite3.Error as e:  # Handle potential SQLite errors
        print(f"An error occurred: {e}")
        
    finally:
        # Ensure the connection exists and is open before closing it.
        # This handles the case if connection fails in the first place.
        if 'conn' in locals() and conn: 
            conn.close()  # Close the database connection