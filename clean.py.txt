data = pd.read_csv(r'C:\Users\anhem44\Desktop\Capstone 2\properties_2016.csv',index_col='parcelid')
y_data = pd.read_csv(r'C:\Users\anhem44\Desktop\Capstone 2\train_2016_v2.csv',index_col='parcelid')


def clean_x(x) :
    #seperate data into frames based off of type
    
    cat_data = x[['architecturalstyletypeid','buildingqualitytypeid','numberofstories']]

    cont_data = x[['structuretaxvaluedollarcnt','censustractandblock','lotsizesquarefeet','finishedsquarefeet50',
                  'finishedsquarefeet15','finishedsquarefeet12','calculatedfinishedsquarefeet','finishedfloor1squarefeet']]

    others = x.drop(x[['architecturalstyletypeid','buildingqualitytypeid','numberofstories',
                         'calculatedbathnbr','structuretaxvaluedollarcnt','censustractandblock','lotsizesquarefeet',
                         'finishedsquarefeet50','finishedsquarefeet15','finishedsquarefeet12','calculatedfinishedsquarefeet',
                         'finishedfloor1squarefeet']], axis = 1)

    others.fillna(0, inplace=True)
    others.head()

    data_list = others.select_dtypes(include=['object']) # list all non-numeric rows
    numeric_only = others._get_numeric_data() #select only numeric rows

    for column in data_list : #make sure all non-numerics are strings
        data_list[column] = data_list[column].astype(str)

    data_list = data_list.apply(preprocessing.LabelEncoder().fit_transform) #encode all nun-numerics using labels
    numeric_only = numeric_only.apply(preprocessing.LabelEncoder().fit_transform)
    data_clean = pd.concat([numeric_only, data_list], axis=1,join_axes=[x.index]) #re-join numeric and non-numeric
    
    return cat_data, cont_data, data_clean

def cat_nan(cat_data,data_clean):
    #use list of names for for loop
    cat_list = list(cat_data) #turn column names into a list
    
    #dfs function will fill and return
    cat_df = pd.DataFrame(index=data_clean.index) #will fill this df w/best prediction answer in a column
    cat_nan_df = pd.DataFrame(columns=['KNN_mean','LGR_mean','RF_mean','best']) #df that will give the score and best 
    
    for i in cat_list :
        # 1- select only those cells w/values for training set
        df_c = pd.concat([data_clean, cat_data[i]],axis=1,join_axes=[data_clean.index]) #join cat data to clean data
        
        df_n_null = df_c[i].notnull() #data that is not null for training set
        df_n_null = df_c[df_n_null]
        df_null = df_c[i].isnull() #data that is null for prediction set
        df_null = df_c[df_null]
        
        y = pd.DataFrame(df_n_null[i]) #target as df
        x = df_n_null.drop(i,axis=1) #remove target from testing set as df
        
        y = y.apply(preprocessing.LabelEncoder().fit_transform) #preproccess target
        
        # 2- Use kfolds for test
        kfold = cross_validation.KFold(len(x),n_folds=5, random_state=42)
        
        # 3- run different tests
        #KNN on test data
        KNN_pred_error = []
        for train, test in kfold :
            x_train, x_test, y_train, y_test = x.iloc[train], x.iloc[test], y.iloc[train], y.iloc[test] 
            KNN = neighbors.KNeighborsClassifier() 
            KNN.fit(x_train,y_train)
            KNN_score = KNN.score(x_test,y_test)
            KNN_pred_error.append(KNN_score)
        KNN_mean = np.mean(KNN_pred_error) #mean of each score
        
        #logistic regression on test data
        LGR_pred_error = []
        for train, test in kfold :
            x_train, x_test, y_train, y_test = x.iloc[train], x.iloc[test], y.iloc[train], y.iloc[test]
            LGR = linear_model.LogisticRegression()
            LGR.fit(x_train,y_train)
            LGR_score = LGR.score(x_test,y_test)
            LGR_pred_error.append(LGR_score)
        LGR_mean = np.mean(LGR_pred_error)
        
        #random forest
        RF_pred_error = []
        for train, test in kfold :
            x_train, x_test, y_train, y_test = x.iloc[train], x.iloc[test], y.iloc[train], y.iloc[test]
            RF = ensemble.RandomForestClassifier()
            RF.fit(x_train,y_train)
            RF_score = RF.score(x_test,y_test)
            RF_pred_error.append(RF_score)
        RF_mean = np.mean(RF_pred_error)
        
        # 4-calculate best predictor
        
        best = str() #will tag best predictor
        if (KNN_mean != LGR_mean) & (KNN_mean != RF_mean) & (LGR_mean != RF_mean) :
            if (KNN_mean > LGR_mean) & (KNN_mean > RF_mean) :
                best = 'KNN' 
            elif (LGR_mean > KNN_mean) & (LGR_mean > RF_mean) :
                best = 'LGR'
            elif (RF_mean > LGR_mean) & (RF_mean > KNN_mean) :
                best = 'RF'
        elif KNN_mean == LGR_mean :
            best = 'LGR' 
        elif KNN_mean == RF_mean :
            best = 'KNN'
        else :
            best = 'LGR'
        
        # add to df
        cat_nan_df.loc[i] = [KNN_mean,LGR_mean,RF_mean,best] #add all means and best to df
        
        # 5- use best predictor to predict missing values for predictor set
        x_null = df_null.drop(i,axis=1)
        
        if best == 'KNN':
            KNN = neighbors.KNeighborsClassifier()
            KNN.fit(x,y)
            prediction = KNN.predict(x_null) 
        elif best == 'LGR':
            LGR = linear_model.LogisticRegression()
            LGR.fit(x,y)
            prediction = LGR.predict(x_null)
        elif best == 'RF':
            RF = ensemble.RandomForestClassifier()
            RF.fit(x,y)
            prediction = RF.predict(x_null)
        
        # 6-insert prediction into data
        predicton_df = pd.DataFrame(index=x_null.index)
        x_total = pd.DataFrame()
        
        predicton_df[i] = prediction
        
        x_total = pd.concat([predicton_df,y]) #combine prediction and test set
        cat_df = pd.concat([cat_df, x_total], axis=1) #add complete column to df
        
    return cat_nan_df, cat_df

#step 3

#select only those cells w/values
def lin_nan(cont_data,data_clean):
    #use list of names for for loop
    lin_list = list(cont_data)
    
    #dfs function will fill and return
    lin_df = pd.DataFrame(index=data_clean.index) #will fill this df w/best prediction answer in a column
    lin_nan_df = pd.DataFrame(columns=['LR_mean','SV_mean','RF_mean','best'])#df that will give the score and best 
    
    for i in lin_list :
        df_c = pd.concat([data_clean, cont_data[i]],axis=1,join_axes=[data_clean.index]) #join cont data to clean data
        
        df_n_null = df_c[i].notnull() #data that is not null for training set
        df_n_null = df_c[df_n_null]
        df_null = df_c[i].isnull() #data that is null for prediction set
        df_null = df_c[df_null]
        
        y = pd.DataFrame(df_n_null[i]) #target as df
        x = df_n_null.drop(i,axis=1) #remove target from testing set as df
        
        y = y.apply(preprocessing.LabelEncoder().fit_transform) #preproccess target
        
        #2- use kfold crossvalidation
        #Use kfolds for test
        kfold = cross_validation.KFold(len(x),n_folds=5, random_state=42)
        
        # 3- run different tests
        #linear regression
        LR_pred_error = []
        for train, test in kfold :
            x_train, x_test, y_train, y_test = x.iloc[train], x.iloc[test], y.iloc[train], y.iloc[test]
            LR = linear_model.LinearRegression()
            LR.fit(x_train,y_train)
            LR_score = LR.score(x_test,y_test)
            LR_pred_error.append(LR_score)
        LR_mean = np.mean(LR_pred_error) #mean of each score
        
        #SVM
        SV_pred_error = []
        for train, test in kfold :
            x_train, x_test, y_train, y_test = x.iloc[train], x.iloc[test], y.iloc[train], y.iloc[test]
            SV = svm.LinearSVR()
            SV.fit(x_train,y_train)
            SV_score = SV.score(x_test,y_test)
            SV_pred_error.append(SV_score)
        SV_mean = np.mean(SV_pred_error)
        
        #random forest
        RF_pred_error = []
        for train, test in kfold :
            x_train, x_test, y_train, y_test = x.iloc[train], x.iloc[test], y.iloc[train], y.iloc[test]
            RF = ensemble.RandomForestRegressor()
            RF.fit(x_train,y_train)
            RF_score = RF.score(x_test,y_test)
            RF_pred_error.append(RF_score)
        RF_mean = np.mean(RF_pred_error)
        
        #4- calculate best predictor
        best = str()
        if (LR_mean != SV_mean) & (LR_mean != RF_mean) & (SV_mean != RF_mean) :
            if (LR_mean > SV_mean) & (LR_mean > RF_mean) :
                best = 'LR'
            elif (SV_mean > LR_mean) & (SV_mean > RF_mean) :
                best = 'SV'
            elif (RF_mean > SV_mean) & (RF_mean > LR_mean) :
                best = 'RF'
        elif LR_mean == SV_mean :
            best = 'SV'
        elif LR_mean == RF_mean :
            best = 'KNN'
        else :
            best = 'SV'
        
        #add to df
        lin_nan_df.loc[i] = [LR_mean,SV_mean,RF_mean,best]
        
        #use best predictor to predict missing values
        x_null = df_null.drop(i,axis=1)
        
        # 5- use best predictor to predict missing values for predictor set
        if best == 'LR':
            LR = linear_model.LinearRegression()
            LR.fit(x,y)
            prediction = LR.predict(x_null) 
        elif best == 'SV':
            SV = svm.LinearSVR()
            SV.fit(x,y)
            prediction = LGR.predict(x_null)
        elif best == 'RF':
            RF = ensemble.RandomForestRegressor()
            RF.fit(x,y)
            prediction = RF.predict(x_null)
        
        #6-insert prediction into data
        predicton_df = pd.DataFrame(index=x_null.index)
        x_total = pd.DataFrame()
        
        predicton_df[i] = prediction
        
        x_total = pd.concat([predicton_df,y])
        lin_df = pd.concat([lin_df, x_total], axis=1)
        
    return lin_nan_df, lin_df

#step 4
def final_comb(data_clean,lin_pred,cat_pred):    
    final_df = pd.concat([data_clean,lin_pred, cat_pred], axis=1) #combine all three
    
    train_data = pd.concat([y_data, final_df], axis=1,join_axes=[y_data.index]) #join on transaction's parcel ID
    train_data['transactiondate'] = pd.to_datetime(train_data['transactiondate']) #make transdate a datetime
    train_data['year'] = train_data['transactiondate'].dt.year #extract year
    train_data['month']  = train_data['transactiondate'].dt.month #extract month
    train_data['parcelid'] = train_data.index #make a coulumn out of the index
    train_data.set_index(['parcelid','transactiondate'], inplace=True) #reset index
    
    return train_data, final_df

cat_data, cont_data, data_clean = clean_x(data)
print("clean done")
lin_info, lin_pred = lin_nan(cont_data,data_clean)
print("lin done")
cat_info, cat_pred = cat_nan(cat_data,data_clean)
print("cat done")
final_train, final_df = final_comb(data_clean,lin_pred,cat_pred)
print("fin done")
final_df.to_csv(r'C:\Users\anhem44\Desktop\Capstone 2\zillow_train.csv')
final_train.to_csv(r'C:\Users\anhem44\Desktop\Capstone 2\zillow_final.csv')
print("export done")