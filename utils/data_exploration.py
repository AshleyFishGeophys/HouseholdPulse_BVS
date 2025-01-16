def find_consv_lib_vs_paid_sick_leave(df_states):
    
    df_new = df_states[['Conservative', 'Liberal', 'Paid sick leave', 'Avg']].copy() 
    
    # Create a new column 'Higher_Conservative'
    df_new['Higher_Conservative'] = df_new['Conservative'] > df_new['Liberal']    
    df_new['Higher_Liberal'] = df_new['Conservative'] < df_new['Liberal']    
    df_new['Equal_Conserv_Liberal'] = df_new['Conservative'] == df_new['Liberal']    

    
    df_higher_conserv_with_paid_sick = df_new[(df_new['Paid sick leave'] == 1) & (df_new['Higher_Conservative'] == True)]
    df_higher_lib_with_paid_sick = df_new[(df_new['Paid sick leave'] == 1) & (df_new['Higher_Liberal'] == True)]
    df_equal_conserv_lib_with_paid_sick = df_new[(df_new['Paid sick leave'] == 1) & (df_new['Equal_Conserv_Liberal'] == True)]
    sick_leave_1 = df_new[(df_new['Paid sick leave'] == 1)]

    columns_to_drop = ['Higher_Conservative', 'Higher_Liberal', 'Equal_Conserv_Liberal'] 

    df_higher_conserv_with_paid_sick.drop(columns=columns_to_drop, axis=1, inplace=True) 
    df_higher_lib_with_paid_sick.drop(columns=columns_to_drop, axis=1, inplace=True) 
    df_equal_conserv_lib_with_paid_sick.drop(columns=columns_to_drop, axis=1, inplace=True) 
    
    print(df_higher_conserv_with_paid_sick)
    print(f'Avg Consv rate: {df_higher_conserv_with_paid_sick["Avg"].mean()}')
    print()
    print(df_higher_lib_with_paid_sick)
    print(f'Avg Lib rate: {df_higher_lib_with_paid_sick["Avg"].mean()}')
    print()
    print(df_equal_conserv_lib_with_paid_sick)
    print(f'Avg Equal rate: {df_equal_conserv_lib_with_paid_sick["Avg"].mean()}') 
    # print()
    # print(sick_leave_1)
    print()

