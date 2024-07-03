import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


#--------------------------------------------------------------------------------------------------------------#

### Clear Screen Function ###
def clearScreen():
    print("\n" * 100)
    
#--------------------------------------------------------------------------------------------------------------#

### Read Data Function ###
def readData():
    matches = []
    with open('data/match_data_v5.csv', 'r') as file:
        # Ignore the first line
        file.readline()
        """
            Matches List:
                Each element in the list is a dictionary with the following keys:
                    gameId -> Id of the match,
                    blueTeam -> Dictionary with the information of the blue team,
                    redTeam -> Dictionary with the information of the red team,
                    firstBlood -> 1 if the blue team got the first blood, 0 otherwise,
                    winner -> 1 if the blue team won, 0 otherwise     
        
        
            Keys of blueTeamInfo:
            
                blueTeamControlWardsPlaced -> Number of control wards placed by the blue team, 
                blueTeamWardsPlaced -> Number of wards placed by the blue team, 
                blueTeamTotalKills -> Total kills of the blue team, 
                blueTeamDragonKills -> Number of dragons killed by the blue team, 
                blueTeamHeraldKills -> Number of heralds killed by the blue team, 
                blueTeamTowersDestroyed -> Number of towers destroyed by the blue team, 
                blueTeamInhibitorsDestroyed -> Number of inhibitors destroyed by the blue team,
                blueTeamTurretPlatesDestroyed -> Number of turret plates destroyed by the blue team,
                blueTeamMinionsKilled -> Number of minions killed by the blue team,
                blueTeamJungleMinions -> Number of jungle minions killed by the blue team,
                blueTeamTotalGold -> Total gold of the blue team,
                blueTeamXp -> Total experience of the blue team,
                blueTeamTotalDamageToChamps -> Total damage to champions of the blue team
    
    
            Keys of redTeamInfo:
            
                redTeamControlWardsPlaced -> Number of control wards placed by the red team, 
                redTeamWardsPlaced -> Number of wards placed by the red team, 
                redTeamTotalKills -> Total kills of the red team, 
                redTeamDragonKills -> Number of dragons killed by the red team, 
                redTeamHeraldKills -> Number of heralds killed by the red team, 
                redTeamTowersDestroyed -> Number of towers destroyed by the red team, 
                redTeamInhibitorsDestroyed -> Number of inhibitors destroyed by the red team,
                redTeamTurretPlatesDestroyed -> Number of turret plates destroyed by the red team,
                redTeamMinionsKilled -> Number of minions killed by the red team,
                redTeamJungleMinions -> Number of jungle minions killed by the red team,
                redTeamTotalGold -> Total gold of the red team,
                redTeamXp -> Total experience of the red team,
                redTeamTotalDamageToChamps -> Total damage to champions of the red team
                
        """
        
        # the csv is structured as follows:
        # gameId,
        # blueTeamControlWardsPlaced,
        # blueTeamWardsPlaced,
        # blueTeamTotalKills,
        # blueTeamDragonKills,
        # blueTeamHeraldKills,
        # blueTeamTowersDestroyed,
        # blueTeamInhibitorsDestroyed,
        # blueTeamTurretPlatesDestroyed,
        # firstBlood,
        # blueTeamMinionsKilled,
        # blueTeamJungleMinions,
        # blueTeamTotalGold,
        # blueTeamXp,
        # blueTeamTotalDamageToChamps,
        # redTeamControlWardsPlaced,
        # redTeamWardsPlaced,
        # redTeamTotalKills,
        # redTeamDragonKills,
        # redTeamHeraldKills,
        # redTeamTowersDestroyed,
        # redTeamInhibitorsDestroyed,
        # redTeamTurretPlatesDestroyed,
        # redTeamMinionsKilled,
        # redTeamJungleMinions,
        # redTeamTotalGold,
        # redTeamXp,
        # redTeamTotalDamageToChamps,
        # winner
        

        
        for line in file:
            data = line.split(',')
            
            match = {
                'gameId': data[0],
                'blueTeam': {
                    'blueTeamControlWardsPlaced': data[1],
                    'blueTeamWardsPlaced': data[2],
                    'blueTeamTotalKills': data[3],
                    'blueTeamDragonKills': data[4],
                    'blueTeamHeraldKills': data[5],
                    'blueTeamTowersDestroyed': data[6],
                    'blueTeamInhibitorsDestroyed': data[7],
                    'blueTeamTurretPlatesDestroyed': data[8],
                    'blueTeamMinionsKilled': data[10],
                    'blueTeamJungleMinions': data[11],
                    'blueTeamTotalGold': data[12],
                    'blueTeamXp': data[13],
                    'blueTeamTotalDamageToChamps': data[14],
                },
                'redTeam': {
                    'redTeamControlWardsPlaced': data[15],
                    'redTeamWardsPlaced': data[16],
                    'redTeamTotalKills': data[17],
                    'redTeamDragonKills': data[18],
                    'redTeamHeraldKills': data[19],
                    'redTeamTowersDestroyed': data[20],
                    'redTeamInhibitorsDestroyed': data[21],
                    'redTeamTurretPlatesDestroyed': data[22],
                    'redTeamMinionsKilled': data[23],
                    'redTeamJungleMinions': data[24],
                    'redTeamTotalGold': data[25],
                    'redTeamXp': data[26],
                    'redTeamTotalDamageToChamps': data[27],
                },
                'firstBlood': data[9],
                'mathcResult': data[28]
                
            }
            
            matches.append(match)
            
    file.close()            
    
    return matches

#--------------------------------------------------------------------------------------------------------------#    

### Predict Match Menu Function ###
def predictMatchMenu(matches, scaler, pca):
    
    clearScreen()
    print("Input all the info separated by commas, field by field or read from a file?")
    print("1. All at once")
    print("2. Field by field")
    print("3. Read from a file")
    print("\n0. Go Back")
    
    
    option = input("\n\nOption: ")
    
    if option == '1':
        clearScreen()
        print("Please enter the following information of the match to predict separated by commas with no spaces:")
        print("blueTeamControlWardsPlaced, blueTeamWardsPlaced, blueTeamTotalKills, blueTeamDragonKills, blueTeamHeraldKills, blueTeamTowersDestroyed, blueTeamInhibitorsDestroyed, blueTeamTurretPlatesDestroyed, firstBlood(1 for blue team, 0 for red team), blueTeamMinionsKilled, blueTeamJungleMinions, blueTeamTotalGold, blueTeamXp, blueTeamTotalDamageToChamps, redTeamControlWardsPlaced, redTeamWardsPlaced, redTeamTotalKills, redTeamDragonKills, redTeamHeraldKills, redTeamTowersDestroyed, redTeamInhibitorsDestroyed, redTeamTurretPlatesDestroyed, redTeamMinionsKilled, redTeamJungleMinions, redTeamTotalGold, redTeamXp, redTeamTotalDamageToChamps")
        
        matchToPredict = input("Match to predict: ").split(',')
        
        match = {
            'blueTeam': {
                'blueTeamControlWardsPlaced': matchToPredict[0],
                'blueTeamWardsPlaced': matchToPredict[1],
                'blueTeamTotalKills': matchToPredict[2],
                'blueTeamDragonKills': matchToPredict[3],
                'blueTeamHeraldKills': matchToPredict[4],
                'blueTeamTowersDestroyed': matchToPredict[5],
                'blueTeamInhibitorsDestroyed': matchToPredict[6],
                'blueTeamTurretPlatesDestroyed': matchToPredict[7],
                'blueTeamMinionsKilled': matchToPredict[9],
                'blueTeamJungleMinions': matchToPredict[10],
                'blueTeamTotalGold': matchToPredict[11],
                'blueTeamXp': matchToPredict[12],
                'blueTeamTotalDamageToChamps': matchToPredict[13],
            },
            'redTeam': {
                'redTeamControlWardsPlaced': matchToPredict[14],
                'redTeamWardsPlaced': matchToPredict[15],
                'redTeamTotalKills': matchToPredict[16],
                'redTeamDragonKills': matchToPredict[17],
                'redTeamHeraldKills': matchToPredict[18],
                'redTeamTowersDestroyed': matchToPredict[19],
                'redTeamInhibitorsDestroyed': matchToPredict[20],
                'redTeamTurretPlatesDestroyed': matchToPredict[21],
                'redTeamMinionsKilled': matchToPredict[22],
                'redTeamJungleMinions': matchToPredict[23],
                'redTeamTotalGold': matchToPredict[24],
                'redTeamXp': matchToPredict[25],
                'redTeamTotalDamageToChamps': matchToPredict[26],
            },
            'firstBlood': matchToPredict[8]
        }
        
        predictStateOfGame(match, scaler)
        
        
    elif option == '2':
        clearScreen()
        print("Please enter the following information of the match to predict:")
            
        matchToPredict = {
            'blueTeam': {
                'blueTeamControlWardsPlaced': input("Number of control wards placed by the blue team: "),
                'blueTeamWardsPlaced': input("Number of wards placed by the blue team: "),
                'blueTeamTotalKills': input("Total kills of the blue team: "),
                'blueTeamDragonKills': input("Number of dragons killed by the blue team: "),
                'blueTeamHeraldKills': input("Number of heralds killed by the blue team: "),
                'blueTeamTowersDestroyed': input("Number of towers destroyed by the blue team: "),
                'blueTeamInhibitorsDestroyed': input("Number of inhibitors destroyed by the blue team: "),
                'blueTeamTurretPlatesDestroyed': input("Number of turret plates destroyed by the blue team: "),
                'blueTeamMinionsKilled': input("Number of minions killed by the blue team: "),
                'blueTeamJungleMinions': input("Number of jungle minions killed by the blue team: "),
                'blueTeamTotalGold': input("Total gold of the blue team: "),
                'blueTeamXp': input("Total experience of the blue team: "),
                'blueTeamTotalDamageToChamps': input("Total damage to champions of the blue team: "),
            },
            'redTeam': {
                'redTeamControlWardsPlaced': input("Number of control wards placed by the red team: "),
                'redTeamWardsPlaced': input("Number of wards placed by the red team: "),
                'redTeamTotalKills': input("Total kills of the red team: "),
                'redTeamDragonKills': input("Number of dragons killed by the red team: "),
                'redTeamHeraldKills': input("Number of heralds killed by the red team: "),
                'redTeamTowersDestroyed': input("Number of towers destroyed by the red team: "),
                'redTeamInhibitorsDestroyed': input("Number of inhibitors destroyed by the red team: "),
                'redTeamTurretPlatesDestroyed': input("Number of turret plates destroyed by the red team: "),
                'redTeamMinionsKilled': input("Number of minions killed by the red team: "),
                'redTeamJungleMinions': input("Number of jungle minions killed by the red team: "),
                'redTeamTotalGold': input("Total gold of the red team: "),
                'redTeamXp': input("Total experience of the red team: "),
                'redTeamTotalDamageToChamps': input("Total damage to champions of the red team: "),
            },
            'firstBlood': input("First blood: 1 if the blue team got the first blood, 0 otherwise: ")
        }
    
        predictStateOfGame(matchToPredict, scaler)

    elif option == '3':
        clearScreen()
        print("The file should be a csv file, inside the predict_data folder, with the following structure:")
        print("27 values separated by commas, with no spaces, representing the match to predict")
        print("The first row should not have any values")
        print("The first value should be blueTeamControlWardsPlaced")
        print("The last value should be redTeamTotalDamageToChamps")

        # enumerate the files in the predict_data folder and ask the user to select one
        files = os.listdir('predict_data/')
        for i, file in enumerate(files):
            print(f"{i+1}. {file}")

        file_option = input("\n\nSelect a file: ")
        chosen_file_name = files[int(file_option) - 1]

        predictStateOfGameFile(chosen_file_name, scaler, pca)

        
    elif option == '0':
        menu()      
    
    else:
        print("Invalid option")
        predictMatchMenu(matches, scaler, pca)
    

### Menu Function ###
def menu():
    matches = readData()
    scaler = StandardScaler()
    X_scaled, y = scaleData(scaler)
    pca = PCA()
    X_pca, y = applyPCA(X_scaled, y, pca)
    
    clearScreen()
    print("Welcome to the League of Legends match predictor")
    print("Please select one of the following options:")
    print("1. Print Data")
    print("2. Visualize Data")
    print("3. Create Models")
    print("4. Predict Match")
    print("\n0. Exit")
    
    option = input("\n\nOption: ")
    
    if option == '1':
        clearScreen()
        for match in matches:
            print("\nMatch: " + match.get('gameId'))
            for key, value in match.get('blueTeam').items():
                print(key + ": " + value)
            for key, value in match.get('redTeam').items():
                print(key + ": " + value)
            
            if match.get('firstBlood') == '1':
                print("First Blood: Blue Team")
            else:
                print("First Blood: Red Team")
                
            if match.get('matchResult') == '1':
                print("Winner: Blue Team")
            else:
                print("Winner: Red Team")
            
        input("\nPress Enter to continue...")
        menu()

    elif option == '2':

        print("Please select one of the following options:")
        print("1. Check for missing values")
        print("2. View Average Value of Each Feature")
        print("3. Check for outliers")
        print("4. Compare model accuracies")
        print("5. Compare model precisions")
        print("6. Compare model recalls")
        print("7. Compare model F1 scores")
        print("8. Compare model confusion matrices")
        print("\n0. Go Back")
        suboption = input("\n\nOption: ")

        if suboption == '1':
            checkMissingValues()
            input("\nPress Enter to continue...")
            menu()

        elif suboption == '2':
            averageValueOfEachFeature()
            input("\nPress Enter to continue...")
            menu()

        elif suboption == '3':
            check_for_outliers()
            input("\nPress Enter to continue...")
            menu()

        elif suboption == '4':
            compare_model_accuracies(X_scaled, y)
            input("\nPress Enter to continue...")
            menu()

        elif suboption == '5':
            compare_model_precision(X_scaled, y)
            input("\nPress Enter to continue...")
            menu()

        elif suboption == '6':
            compare_model_recall(X_scaled, y)
            input("\nPress Enter to continue...")
            menu()

        elif suboption == '7':
            compare_model_f1(X_scaled, y)
            input("\nPress Enter to continue...")
            menu()

        elif suboption == '8':
            compare_model_confusion_matrices(X_scaled, y)
            input("\nPress Enter to continue...")
            menu()

        elif suboption == '0':
            menu()

        else:
            print("Invalid option")
            menu()
                
    elif option == '3':

        print("Please select one of the following options:")
        print("1. Decision Tree Classifier")
        print("2. K-Nearest Neighbors")
        print("3. Support Vector Machine")
        print("4. Naive Bayes")
        print("5. Random Forest")
        print("6. Logistic Regression")
        print("7. Neural Network")
        print("\n0. Go Back")
        suboption = input("\n\nOption: ")

        if suboption == '1':
            print("Decision Tree Classifier")
            print("Please select one of the following options:")
            print("1. Insert one value to split the data into training and testing sets")
            print("2. Run for different splits of the data between 0.1/0.9 and 0.9/0.1")
            subsuboption = input("\n\nOption: ")

            if subsuboption == '1':
                test_size = (input("\nInsert a value to split the data into training and testing sets (default is 0.1 - 0.9 training/0.1 testing): "))
                if test_size == '':
                    test_size = 0.1
                else:
                    test_size = float(test_size)
                create_decision_tree_scaled(X_scaled, y, test_size)
                create_decision_tree_PCA(X_pca, y, test_size)
                input("\nPress Enter to continue...")
                menu()

            elif subsuboption == '2':
                # Initialize lists to store the accuracies
                accuracies_scaled = []
                accuracies_PCA = []
                training_sizes = []

                for i in range(1, 10):
                    test_size = i / 10
                    training_size = 1 - test_size
                    training_sizes.append(training_size)
                    print(f"\nTraining size: {training_size}")

                    # Store the accuracy of each model
                    _, accuracy_scaled = create_decision_tree_scaled(X_scaled, y, test_size)
                    _, accuracy_PCA = create_decision_tree_PCA(X_pca, y, test_size)
                    accuracies_scaled.append(accuracy_scaled)
                    accuracies_PCA.append(accuracy_PCA)

                # Plot the accuracies
                plt.plot(training_sizes, accuracies_scaled, label='Scaled')
                plt.plot(training_sizes, accuracies_PCA, label='PCA')
                plt.xlabel('Training Size')
                plt.ylabel('Accuracy')
                plt.title('Accuracy of Decision Tree Models')
                plt.legend()
                plt.show()

                input("\nPress Enter to continue...")
                menu()
            
            else:
                print("Invalid option")
                menu()

        elif suboption == '2':
            print("K-Nearest Neighbors")
            print("Please select one of the following options:")
            print("1. Insert one value to split the data into training and testing sets")
            print("2. Run for different splits of the data between 0.1/0.9 and 0.9/0.1")
            subsuboption = input("\n\nOption: ")
            
            if subsuboption == '1':
                test_size = (input("\nInsert a value to split the data into training and testing sets (default is 0.1 - 0.9 training/0.1 testing): "))
                if test_size == '':
                    test_size = 0.1
                else:
                    test_size = float(test_size)
                create_knn_model(X_scaled, y, test_size)
                create_knn_model_PCA(X_pca, y, test_size)
                input("\nPress Enter to continue...")
                menu()

            elif subsuboption == '2':
                # Initialize lists to store the accuracies
                accuracies_scaled = []
                accuracies_PCA = []
                training_sizes = []

                for i in range(1, 10):
                    test_size = i / 10
                    training_size = 1 - test_size
                    training_sizes.append(training_size)
                    print(f"\nTraining size: {training_size}")

                    # Store the accuracy of each model
                    _, accuracy_scaled = create_knn_model(X_scaled, y, test_size)
                    _, accuracy_PCA = create_knn_model_PCA(X_pca, y, test_size)
                    accuracies_scaled.append(accuracy_scaled)
                    accuracies_PCA.append(accuracy_PCA)

                # Plot the accuracies
                plt.plot(training_sizes, accuracies_scaled, label='Scaled')
                plt.plot(training_sizes, accuracies_PCA, label='PCA')
                plt.xlabel('Training Size')
                plt.ylabel('Accuracy')
                plt.title('Accuracy of K-Nearest Neighbors Models')
                plt.legend()
                plt.show()

                input("\nPress Enter to continue...")
                menu()

            else:
                print("Invalid option")
                menu()

        elif suboption == '3':
            print("Support Vector Machine")
            print("Please select one of the following options:")
            print("1. Insert one value to split the data into training and testing sets")
            print("2. Run for different splits of the data between 0.1/0.9 and 0.9/0.1")

            subsuboption = input("\n\nOption: ")

            if subsuboption == '1':
                test_size = (input("\nInsert a value to split the data into training and testing sets (default is 0.1 - 0.9 training/0.1 testing): "))
                if test_size == '':
                    test_size = 0.1
                else:
                    test_size = float(test_size)
                create_svm_model(X_scaled, y, test_size)
                create_svm_model_PCA(X_pca, y, test_size)
                input("\nPress Enter to continue...")
                menu()

            elif subsuboption == '2':
                # Initialize lists to store the accuracies
                accuracies_scaled = []
                accuracies_PCA = []
                training_sizes = []

                for i in range(1, 10):
                    test_size = i / 10
                    training_size = 1 - test_size
                    training_sizes.append(training_size)
                    print(f"\nTraining size: {training_size}")

                    # Store the accuracy of each model
                    _, accuracy_scaled = create_svm_model(X_scaled, y, test_size)
                    _, accuracy_PCA = create_svm_model_PCA(X_pca, y, test_size)
                    accuracies_scaled.append(accuracy_scaled)
                    accuracies_PCA.append(accuracy_PCA)

                # Plot the accuracies
                plt.plot(training_sizes, accuracies_scaled, label='Scaled')
                plt.plot(training_sizes, accuracies_PCA, label='PCA')
                plt.xlabel('Training Size')
                plt.ylabel('Accuracy')
                plt.title('Accuracy of Support Vector Machine Models')
                plt.legend()
                plt.show()

                input("\nPress Enter to continue...")
                menu()

            else:
                print("Invalid option")
                menu()

        elif suboption == '4':
            print("Naive Bayes")
            print("Please select one of the following options:")
            print("1. Insert one value to split the data into training and testing sets")
            print("2. Run for different splits of the data between 0.1/0.9 and 0.9/0.1")

            subsuboption = input("\n\nOption: ")

            if subsuboption == '1':
                test_size = (input("\nInsert a value to split the data into training and testing sets (default is 0.1 - 0.9 training/0.1 testing): "))
                if test_size == '':
                    test_size = 0.1
                else:
                    test_size = float(test_size)
                create_naive_bayes_model(X_scaled, y, test_size)
                create_naive_bayes_model_PCA(X_pca, y, test_size)
                input("\nPress Enter to continue...")
                menu()

            elif subsuboption == '2':
                # Initialize lists to store the accuracies
                accuracies_scaled = []
                accuracies_PCA = []
                training_sizes = []

                for i in range(1, 10):
                    test_size = i / 10
                    training_size = 1 - test_size
                    training_sizes.append(training_size)
                    print(f"\nTraining size: {training_size}")

                    # Store the accuracy of each model
                    _, accuracy_scaled = create_naive_bayes_model(X_scaled, y, test_size)
                    _, accuracy_PCA = create_naive_bayes_model_PCA(X_pca, y, test_size)
                    accuracies_scaled.append(accuracy_scaled)
                    accuracies_PCA.append(accuracy_PCA)

                # Plot the accuracies
                plt.plot(training_sizes, accuracies_scaled, label='Scaled')
                plt.plot(training_sizes, accuracies_PCA, label='PCA')
                plt.xlabel('Training Size')
                plt.ylabel('Accuracy')
                plt.title('Accuracy of Naive Bayes Models')
                plt.legend()
                plt.show()

                input("\nPress Enter to continue...")
                menu()

            else:
                print("Invalid option")
                menu()

        elif suboption == '5':
            print("Random Forest")
            print("Please select one of the following options:")
            print("1. Insert one value to split the data into training and testing sets")
            print("2. Run for different splits of the data between 0.1/0.9 and 0.9/0.1")

            subsuboption = input("\n\nOption: ")

            if subsuboption == '1':
                test_size = (input("\nInsert a value to split the data into training and testing sets (default is 0.1 - 0.9 training/0.1 testing): "))
                if test_size == '':
                    test_size = 0.1
                else:
                    test_size = float(test_size)
                create_random_forest_model(X_scaled, y, test_size)
                create_random_forest_model_PCA(X_pca, y, test_size)
                input("\nPress Enter to continue...")
                menu()

            elif subsuboption == '2':
                # Initialize lists to store the accuracies
                accuracies_scaled = []
                accuracies_PCA = []
                training_sizes = []

                for i in range(1, 10):
                    test_size = i / 10
                    training_size = 1 - test_size
                    training_sizes.append(training_size)
                    print(f"\nTraining size: {training_size}")

                    # Store the accuracy of each model
                    _, accuracy_scaled = create_random_forest_model(X_scaled, y, test_size)
                    _, accuracy_PCA = create_random_forest_model_PCA(X_pca, y, test_size)
                    accuracies_scaled.append(accuracy_scaled)
                    accuracies_PCA.append(accuracy_PCA)

                # Plot the accuracies
                plt.plot(training_sizes, accuracies_scaled, label='Scaled')
                plt.plot(training_sizes, accuracies_PCA, label='PCA')
                plt.xlabel('Training Size')
                plt.ylabel('Accuracy')
                plt.title('Accuracy of Random Forest Models')
                plt.legend()
                plt.show()

                input("\nPress Enter to continue...")
                menu()

            else:
                print("Invalid option")
                menu()

        elif suboption == '6':
            print("Logistic Regression")
            print("Please select one of the following options:")
            print("1. Insert one value to split the data into training and testing sets")
            print("2. Run for different splits of the data between 0.1/0.9 and 0.9/0.1")

            subsuboption = input("\n\nOption: ")

            if subsuboption == '1':
                test_size = (input("\nInsert a value to split the data into training and testing sets (default is 0.1 - 0.9 training/0.1 testing): "))
                if test_size == '':
                    test_size = 0.1
                else:
                    test_size = float(test_size)
                create_logistic_regression_model(X_scaled, y, test_size)
                create_logistic_regression_model_PCA(X_pca, y, test_size)
                input("\nPress Enter to continue...")
                menu()

            elif subsuboption == '2':
                # Initialize lists to store the accuracies
                accuracies_scaled = []
                accuracies_PCA = []
                training_sizes = []

                for i in range(1, 10):
                    test_size = i / 10
                    training_size = 1 - test_size
                    training_sizes.append(training_size)
                    print(f"\nTraining size: {training_size}")

                    # Store the accuracy of each model
                    _, accuracy_scaled = create_logistic_regression_model(X_scaled, y, test_size)
                    _, accuracy_PCA = create_logistic_regression_model_PCA(X_pca, y, test_size)
                    accuracies_scaled.append(accuracy_scaled)
                    accuracies_PCA.append(accuracy_PCA)

                # Plot the accuracies
                plt.plot(training_sizes, accuracies_scaled, label='Scaled')
                plt.plot(training_sizes, accuracies_PCA, label='PCA')
                plt.xlabel('Training Size')
                plt.ylabel('Accuracy')
                plt.title('Accuracy of Logistic Regression Models')
                plt.legend()
                plt.show()

                input("\nPress Enter to continue...")
                menu()

            else:
                print("Invalid option")
                menu()

        elif suboption == '7':
            print("Neural Network")
            print("Please select one of the following options:")
            print("1. Insert one value to split the data into training and testing sets")
            print("2. Run for different splits of the data between 0.1/0.9 and 0.9/0.1")

            subsuboption = input("\n\nOption: ")

            if subsuboption == '1':
                test_size = (input("\nInsert a value to split the data into training and testing sets (default is 0.1 - 0.9 training/0.1 testing): "))
                if test_size == '':
                    test_size = 0.1
                else:
                    test_size = float(test_size)
                create_neural_network_model(X_scaled, y, test_size)
                create_neural_network_model_PCA(X_pca, y, test_size)
                input("\nPress Enter to continue...")
                menu()

            elif subsuboption == '2':
                # Initialize lists to store the accuracies
                accuracies_scaled = []
                accuracies_PCA = []
                training_sizes = []

                for i in range(1, 10):
                    test_size = i / 10
                    training_size = 1 - test_size
                    training_sizes.append(training_size)
                    print(f"\nTraining size: {training_size}")

                    # Store the accuracy of each model
                    _, accuracy_scaled = create_neural_network_model(X_scaled, y, test_size)
                    _, accuracy_PCA = create_neural_network_model_PCA(X_pca, y, test_size)
                    accuracies_scaled.append(accuracy_scaled)
                    accuracies_PCA.append(accuracy_PCA)

                # Plot the accuracies
                plt.plot(training_sizes, accuracies_scaled, label='Scaled')
                plt.plot(training_sizes, accuracies_PCA, label='PCA')
                plt.xlabel('Training Size')
                plt.ylabel('Accuracy')
                plt.title('Accuracy of Neural Network Models')
                plt.legend()
                plt.show()

                input("\nPress Enter to continue...")
                menu()

            else:
                print("Invalid option")
                menu()

        elif suboption == '0':
            menu()

        else:
            print("Invalid option")
            menu()

    elif option == '4':
        predictMatchMenu(matches, scaler, pca)
        
    elif option == '0':
        sys.exit()
        
    else:
        print("Invalid option")
        menu()

#--------------------------------------------------------------------------------------------------------------#

### Data Visualization ###

def checkMissingValues():
    """
        This function checks if there are any missing values in the data.
    """
    with open('data/match_data_v5.csv', 'r') as file:
        # Ignore the first line
        file.readline()
        
        for line in file:
            data = line.split(',')
            
            for value in data:
                if value == '':
                    print("Missing value found")
                    return
                
    print("No missing values found")
    
    file.close()


def averageValueOfEachFeature():
    """
        This function calculates the average value of each feature in the data
        and displays a bar plot comparing the average value of each feature using logarithmic scaling.
    """
    with open('data/match_data_v5.csv', 'r') as file:
        # Ignore the first line
        file.readline()
        
        # Initialize the variables to store the sum of each feature
        blueTeamControlWardsPlaced = 0
        blueTeamWardsPlaced = 0
        blueTeamTotalKills = 0
        blueTeamDragonKills = 0
        blueTeamHeraldKills = 0
        blueTeamTowersDestroyed = 0
        blueTeamInhibitorsDestroyed = 0
        blueTeamTurretPlatesDestroyed = 0
        blueTeamMinionsKilled = 0
        blueTeamJungleMinions = 0
        blueTeamTotalGold = 0
        blueTeamXp = 0
        blueTeamTotalDamageToChamps = 0
        
        redTeamControlWardsPlaced = 0
        redTeamWardsPlaced = 0
        redTeamTotalKills = 0
        redTeamDragonKills = 0
        redTeamHeraldKills = 0
        redTeamTowersDestroyed = 0
        redTeamInhibitorsDestroyed = 0
        redTeamTurretPlatesDestroyed = 0
        redTeamMinionsKilled = 0
        redTeamJungleMinions = 0
        redTeamTotalGold = 0
        redTeamXp = 0
        redTeamTotalDamageToChamps = 0
        
        total_matches = 0

        for line in file:
            data = line.split(',')
            total_matches += 1
            
            blueTeamControlWardsPlaced += int(data[1])
            blueTeamWardsPlaced += int(data[2])
            blueTeamTotalKills += int(data[3])
            blueTeamDragonKills += int(data[4])
            blueTeamHeraldKills += int(data[5])
            blueTeamTowersDestroyed += int(data[6])
            blueTeamInhibitorsDestroyed += int(data[7])
            blueTeamTurretPlatesDestroyed += int(data[8])
            blueTeamMinionsKilled += int(data[10])
            blueTeamJungleMinions += int(data[11])
            blueTeamTotalGold += int(data[12])
            blueTeamXp += int(data[13])
            blueTeamTotalDamageToChamps += int(data[14])

            redTeamControlWardsPlaced += int(data[15])
            redTeamWardsPlaced += int(data[16])
            redTeamTotalKills += int(data[17])
            redTeamDragonKills += int(data[18])
            redTeamHeraldKills += int(data[19])
            redTeamTowersDestroyed += int(data[20])
            redTeamInhibitorsDestroyed += int(data[21])
            redTeamTurretPlatesDestroyed += int(data[22])
            redTeamMinionsKilled += int(data[23])
            redTeamJungleMinions += int(data[24])
            redTeamTotalGold += int(data[25])
            redTeamXp += int(data[26])
            redTeamTotalDamageToChamps += int(data[27])

        # Calculate the average for each feature
        blue_averages = [
            blueTeamControlWardsPlaced / total_matches,
            blueTeamWardsPlaced / total_matches,
            blueTeamTotalKills / total_matches,
            blueTeamDragonKills / total_matches,
            blueTeamHeraldKills / total_matches,
            blueTeamTowersDestroyed / total_matches,
            blueTeamInhibitorsDestroyed / total_matches,
            blueTeamTurretPlatesDestroyed / total_matches,
            blueTeamMinionsKilled / total_matches,
            blueTeamJungleMinions / total_matches,
            blueTeamTotalGold / total_matches,
            blueTeamXp / total_matches,
            blueTeamTotalDamageToChamps / total_matches
        ]
        
        red_averages = [
            redTeamControlWardsPlaced / total_matches,
            redTeamWardsPlaced / total_matches,
            redTeamTotalKills / total_matches,
            redTeamDragonKills / total_matches,
            redTeamHeraldKills / total_matches,
            redTeamTowersDestroyed / total_matches,
            redTeamInhibitorsDestroyed / total_matches,
            redTeamTurretPlatesDestroyed / total_matches,
            redTeamMinionsKilled / total_matches,
            redTeamJungleMinions / total_matches,
            redTeamTotalGold / total_matches,
            redTeamXp / total_matches,
            redTeamTotalDamageToChamps / total_matches
        ]

        features = [
            'Control Wards Placed', 'Wards Placed', 'Total Kills', 'Dragon Kills', 'Herald Kills',
            'Towers Destroyed', 'Inhibitors Destroyed', 'Turret Plates Destroyed', 'Minions Killed',
            'Jungle Minions', 'Total Gold', 'XP', 'Total Damage To Champs'
        ]

        # Apply logarithmic scaling to the average values
        blue_averages_log = [np.log1p(x) for x in blue_averages]
        red_averages_log = [np.log1p(x) for x in red_averages]

        # Plot the average values with logarithmic scaling
        x = range(len(features))
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.bar(x, blue_averages_log, width=0.4, label='Blue Team', align='center')
        ax.bar(x, red_averages_log, width=0.4, label='Red Team', align='edge')
        
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.set_ylabel('Logarithmic Average Value')
        ax.set_title('Logarithmic Average Value of Each Feature for Blue and Red Teams')
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    file.close()

def check_for_outliers():
    """
    This function displays a scatter plot for each feature in the data to check for outliers.
    """
    # Define the feature names (excluding the 'winner' column)
    feature_names = [
        'blueTeamControlWardsPlaced',
        'blueTeamWardsPlaced',
        'blueTeamTotalKills',
        'blueTeamDragonKills',
        'blueTeamHeraldKills',
        'blueTeamTowersDestroyed',
        'blueTeamInhibitorsDestroyed',
        'blueTeamTurretPlatesDestroyed',
        'firstBlood',
        'blueTeamMinionsKilled',
        'blueTeamJungleMinions',
        'blueTeamTotalGold',
        'blueTeamXp',
        'blueTeamTotalDamageToChamps',
        'redTeamControlWardsPlaced',
        'redTeamWardsPlaced',
        'redTeamTotalKills',
        'redTeamDragonKills',
        'redTeamHeraldKills',
        'redTeamTowersDestroyed',
        'redTeamInhibitorsDestroyed',
        'redTeamTurretPlatesDestroyed',
        'redTeamMinionsKilled',
        'redTeamJungleMinions',
        'redTeamTotalGold',
        'redTeamXp',
        'redTeamTotalDamageToChamps'
    ]
    
    # Load the data into a pandas DataFrame
    data = pd.read_csv('data/match_data_v5.csv', names=[
        'gameId',
        'blueTeamControlWardsPlaced',
        'blueTeamWardsPlaced',
        'blueTeamTotalKills',
        'blueTeamDragonKills',
        'blueTeamHeraldKills',
        'blueTeamTowersDestroyed',
        'blueTeamInhibitorsDestroyed',
        'blueTeamTurretPlatesDestroyed',
        'firstBlood',
        'blueTeamMinionsKilled',
        'blueTeamJungleMinions',
        'blueTeamTotalGold',
        'blueTeamXp',
        'blueTeamTotalDamageToChamps',
        'redTeamControlWardsPlaced',
        'redTeamWardsPlaced',
        'redTeamTotalKills',
        'redTeamDragonKills',
        'redTeamHeraldKills',
        'redTeamTowersDestroyed',
        'redTeamInhibitorsDestroyed',
        'redTeamTurretPlatesDestroyed',
        'redTeamMinionsKilled',
        'redTeamJungleMinions',
        'redTeamTotalGold',
        'redTeamXp',
        'redTeamTotalDamageToChamps',
        'winner'
    ])

    # Reset the index to ensure it's numeric
    data.reset_index(drop=True, inplace=True)

    # Create scatter plots for each feature
    for feature in feature_names:
        plt.figure(figsize=(10, 6))
        plt.scatter(data.index, data[feature])
        plt.title(f'Scatter Plot for {feature}')
        plt.xlabel('Index')
        plt.ylabel(feature)
        plt.grid(True)
        plt.show()

def compare_model_accuracies(X_scaled, y):
    """
    This function compares the accuracies of the different models and plots the results.
    """
    # Get the models in the models folder
    models = os.listdir('models/')

    #if no models are found, return
    if len(models) == 0:
        print("No models found")
        sys.exit()

    # Initialize lists to store the accuracies and model names
    accuracies = []
    model_names = []

    for model in models:
        model_name=model.split('.')[0]
        model_names.append(model_name)

        if 'PCA' not in model:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

            # Load the model
            loaded_model = pickle.load(open(f'models/{model}', 'rb'))

            # Evaluate the model
            accuracy = loaded_model.score(X_test, y_test) * 100  # Convert to percentage
            accuracies.append(accuracy)
        else:
            pca = PCA(n_components=17)
            X_pca = pca.fit_transform(X_scaled)
            X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.1, random_state=42)

            # Load the model
            loaded_model = pickle.load(open(f'models/{model}', 'rb'))

            # Evaluate the model
            accuracy = loaded_model.score(X_test, y_test) * 100  # Convert to percentage
            accuracies.append(accuracy)

    # Plot the accuracies
    x = range(len(model_names))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x, accuracies, width=0.4, color='blue', align='center')

    # Set the x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracies Comparison')

    plt.tight_layout()
    plt.show()

def compare_model_precision(X_scaled, y):
    """
    This function compares the precision of the different models and plots the results.
    """
    # Get the models in the models folder
    models = os.listdir('models/')

    #if no models are found, return
    if len(models) == 0:
        print("No models found")
        sys.exit()

    # Initialize lists to store the accuracies and model names
    precisions = []
    model_names = []

    for model in models:
        model_name=model.split('.')[0]
        model_names.append(model_name)

        if 'PCA' not in model:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

            # Load the model
            loaded_model = pickle.load(open(f'models/{model}', 'rb'))

            # Evaluate the model
            y_pred = loaded_model.predict(X_test)
            precision = precision_score(y_test, y_pred, average='weighted')
            precisions.append(precision)
        else:
            pca = PCA(n_components=17)
            X_pca = pca.fit_transform(X_scaled)
            X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.1, random_state=42)

            # Load the model
            loaded_model = pickle.load(open(f'models/{model}', 'rb'))

            # Evaluate the model
            y_pred = loaded_model.predict(X_test)
            precision = precision_score(y_test, y_pred, average='weighted')
            precisions.append(precision)

    # Plot the accuracies
    x = range(len(model_names))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x, precisions, width=0.4, color='blue', align='center')

    # Set the x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Precision')
    ax.set_title('Model Precision Comparison')

    plt.tight_layout()
    plt.show()

def compare_model_recall(X_scaled, y):
    """
    This function compares the recall of the different models and plots the results.
    """
    # Get the models in the models folder
    models = os.listdir('models/')

    #if no models are found, return
    if len(models) == 0:
        print("No models found")
        sys.exit()

    # Initialize lists to store the accuracies and model names
    recalls = []
    model_names = []

    for model in models:
        model_name=model.split('.')[0]
        model_names.append(model_name)

        if 'PCA' not in model:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

            # Load the model
            loaded_model = pickle.load(open(f'models/{model}', 'rb'))

            # Evaluate the model
            y_pred = loaded_model.predict(X_test)
            recall = recall_score(y_test, y_pred, average='weighted')
            recalls.append(recall)
        else:
            pca = PCA(n_components=17)
            X_pca = pca.fit_transform(X_scaled)
            X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.1, random_state=42)

            # Load the model
            loaded_model = pickle.load(open(f'models/{model}', 'rb'))

            # Evaluate the model
            y_pred = loaded_model.predict(X_test)
            recall = recall_score(y_test, y_pred, average='weighted')
            recalls.append(recall)

    # Plot the accuracies
    x = range(len(model_names))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x, recalls, width=0.4, color='blue', align='center')

    # Set the x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Recall')
    ax.set_title('Model Recall Comparison')

    plt.tight_layout()
    plt.show()

def compare_model_f1(X_scaled, y):
    """
    This function compares the f1 score of the different models and plots the results.
    """
    # Get the models in the models folder
    models = os.listdir('models/')

    #if no models are found, return
    if len(models) == 0:
        print("No models found")
        sys.exit()

    # Initialize lists to store the accuracies and model names
    f1_scores = []
    model_names = []

    for model in models:
        model_name=model.split('.')[0]
        model_names.append(model_name)

        if 'PCA' not in model:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

            # Load the model
            loaded_model = pickle.load(open(f'models/{model}', 'rb'))

            # Evaluate the model
            y_pred = loaded_model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            f1_scores.append(f1)
        else:
            pca = PCA(n_components=17)
            X_pca = pca.fit_transform(X_scaled)
            X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.1, random_state=42)

            # Load the model
            loaded_model = pickle.load(open(f'models/{model}', 'rb'))

            # Evaluate the model
            y_pred = loaded_model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            f1_scores.append(f1)

    # Plot the accuracies
    x = range(len(model_names))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x, f1_scores, width=0.4, color='blue', align='center')

    # Set the x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('F1 Score')
    ax.set_title('Model F1 Score Comparison')

    plt.tight_layout()
    plt.show()

def compare_model_confusion_matrices(X_scaled, y):
    """
    This function compares the confusion matrices of the different models and writes the results to a text file.
    """
    # Get the models in the models folder
    models = os.listdir('models/')

    #if no models are found, return
    if len(models) == 0:
        print("No models found")
        sys.exit()

    # Open the text file
    with open('matrix/matrix.txt', 'w') as f:
        for model in models:
            if 'PCA' not in model:
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

                # Load the model
                loaded_model = pickle.load(open(f'models/{model}', 'rb'))

                # Evaluate the model
                y_pred = loaded_model.predict(X_test)

                # Create the confusion matrix
                cm = confusion_matrix(y_test, y_pred)

                # Write the confusion matrix to the text file
                f.write(f"Model Name: {model}\n")
                f.write(f"TP: {cm[0][0]}, FP: {cm[0][1]}\n")
                f.write(f"FN: {cm[1][0]}, TN: {cm[1][1]}\n\n")

            else:
                pca = PCA(n_components=17)
                X_pca = pca.fit_transform(X_scaled)
                X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.1, random_state=42)

                # Load the model
                loaded_model = pickle.load(open(f'models/{model}', 'rb'))

                # Evaluate the model
                y_pred = loaded_model.predict(X_test)

                # Create the confusion matrix
                cm = confusion_matrix(y_test, y_pred)

                # Write the confusion matrix to the text file
                f.write(f"Model Name: {model}\n")
                f.write(f"TP: {cm[0][0]}, FP: {cm[0][1]}\n")
                f.write(f"FN: {cm[1][0]}, TN: {cm[1][1]}\n\n")
    
    print("Confusion matrices written to matrix/matrix.txt")

#--------------------------------------------------------------------------------------------------------------#

### Data Transformation ###

# Before training the model, we should scale the data to ensure that all features contribute equally to the prediction.

def scaleData(scaler, dataset='data/match_data_v5.csv'):
    """
    This function scales the data using the StandardScaler and returns the scaled features and the target.
    """

    column_names = [
    'gameId', # This is not a feature
    'blueTeamControlWardsPlaced',
    'blueTeamWardsPlaced',
    'blueTeamTotalKills',
    'blueTeamDragonKills',
    'blueTeamHeraldKills',
    'blueTeamTowersDestroyed',
    'blueTeamInhibitorsDestroyed',
    'blueTeamTurretPlatesDestroyed',
    'firstBlood',
    'blueTeamMinionsKilled',
    'blueTeamJungleMinions',
    'blueTeamTotalGold',
    'blueTeamXp',
    'blueTeamTotalDamageToChamps',
    'redTeamControlWardsPlaced',
    'redTeamWardsPlaced',
    'redTeamTotalKills',
    'redTeamDragonKills',
    'redTeamHeraldKills',
    'redTeamTowersDestroyed',
    'redTeamInhibitorsDestroyed',
    'redTeamTurretPlatesDestroyed',
    'redTeamMinionsKilled',
    'redTeamJungleMinions',
    'redTeamTotalGold',
    'redTeamXp',
    'redTeamTotalDamageToChamps',
    'winner', # This is the target
    'filler'  
]
    # Load the data
    data = pd.read_csv(dataset, names = column_names, skiprows=1)

    # Drop the first and last columns
    data.drop(data.columns[[0, -1]], axis=1, inplace=True)

    # Split the data into features and target
    X = data.iloc[:, :-1]  # All columns except the last one
    y = data.iloc[:, -1]   # Only the penultimate column

    """
    print("Features:", X.columns)
    print("Values:", X.values)

    print("Target:", y.name)
    print("Values:", y.values)
    """

    # Ensure that y does not contain NaN values
    if y.isnull().any():
        print("Warning: NaN values found in target column")
        y = y.fillna(0)

    # Scale the features
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def determine_pca_components(X_scaled, y, pca, dataset='data/match_data_v5.csv'):
    """
    This function determines the number of principal components that explain at least 90% of the variance.
    """

    # Fit PCA
    
    pca.fit(X_scaled)

    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()

    """
    # Plot the explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='o', linestyle='--')
    plt.title('Cumulative Explained Variance by Number of Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()
    plt.show()
    """

    # Find the number of components that explain at least 90% of the variance
    n_components_90 = (cumulative_explained_variance >= 0.90).argmax() + 1
    print(f'Number of components that explain at least 90% of the variance: {n_components_90}')

    return n_components_90

def applyPCA(X_scaled, y, pca, dataset='data/match_data_v5.csv'):
    """
    This function applies PCA to reduce the number of features and returns the transformed features and the target.
    """

    n_components = determine_pca_components(X_scaled, y, pca, dataset)

    # Apply PCA to reduce the number of features
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    #print("PCA components:", pca.components_)
    #print("PCA explained variance ratio:", pca.explained_variance_ratio_)

    return X_pca, y


#--------------------------------------------------------------------------------------------------------------#

### Decision Tree Classifier ###

def create_decision_tree_scaled(X_scaled, y, test_size=0.1):

    """
    This function creates a decision tree classifier using the scaled features and evaluates its accuracy.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    # Create a decision tree classifier
    decision_tree = DecisionTreeClassifier(random_state=42)

    # Fit the decision tree classifier to the training data
    decision_tree.fit(X_train, y_train)

    # Evaluate the decision tree classifier
    accuracy = decision_tree.score(X_test, y_test)
    print("Decision Tree Accuracy:", accuracy)

    # Remove the . from the test_size
    test_size = str(test_size).replace('.', '_')

    # Save the model to disk
    filename = f'models/decisionTreeScaled{test_size}.sav'
    pickle.dump(decision_tree, open(filename, 'wb'))

    return decision_tree, accuracy

def create_decision_tree_PCA(X_pca, y, test_size=0.1):

    """
    This function creates a decision tree classifier using the PCA features and evaluates its accuracy.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, random_state=42)

    # Create a decision tree classifier
    decision_tree = DecisionTreeClassifier(random_state=42)

    # Fit the decision tree classifier to the training data
    decision_tree.fit(X_train, y_train)

    # Evaluate the decision tree classifier
    accuracy = decision_tree.score(X_test, y_test)
    print("Decision Tree Accuracy with PCA:", accuracy)

    # Remove the . from the test_size
    test_size = str(test_size).replace('.', '_')

    # Save the model to disk
    filename = f'models/decisionTreePCA{test_size}.sav'
    pickle.dump(decision_tree, open(filename, 'wb'))

    return decision_tree, accuracy

#--------------------------------------------------------------------------------------------------------------#

### K-Nearest Neighbors ###

def create_knn_model(X_scaled, y, test_size=0.1):

    """
    This function creates a K-Nearest Neighbors classifier using the scaled features and evaluates its accuracy.
    """
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    
    # Create a K-Nearest Neighbors classifier
    knn = KNeighborsClassifier()
    
    # Fit the K-Nearest Neighbors classifier to the training data
    knn.fit(X_train, y_train)
    
    # Evaluate the K-Nearest Neighbors classifier
    accuracy = knn.score(X_test, y_test)
    print("K-Nearest Neighbors Accuracy:", accuracy)

    # Remove the . from the test_size
    test_size = str(test_size).replace('.', '_')

    # Save the model to disk
    filename = f'models/knnScaled{test_size}.sav'
    pickle.dump(knn, open(filename, 'wb'))
    
    return knn, accuracy

def create_knn_model_PCA(X_pca, y, test_size=0.1):

    """
    This function creates a K-Nearest Neighbors classifier using the PCA features and evaluates its accuracy.
    """
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, random_state=42)
    
    # Create a K-Nearest Neighbors classifier
    knn = KNeighborsClassifier()
    
    # Fit the K-Nearest Neighbors classifier to the training data
    knn.fit(X_train, y_train)
    
    # Evaluate the K-Nearest Neighbors classifier
    accuracy = knn.score(X_test, y_test)
    print("K-Nearest Neighbors Accuracy with PCA:", accuracy)

    # Remove the . from the test_size
    test_size = str(test_size).replace('.', '_')

    # Save the model to disk
    filename = f'models/knnPCA{test_size}.sav'
    pickle.dump(knn, open(filename, 'wb'))
    
    return knn, accuracy

#--------------------------------------------------------------------------------------------------------------#

### Support Vector Machine ###

def create_svm_model(X_scaled, y, test_size=0.1):

    """
    This function creates a Support Vector Machine classifier using the scaled features and evaluates its accuracy.
    """
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    
    # Create a Support Vector Machine classifier
    svm = SVC()
    
    # Fit the Support Vector Machine classifier to the training data
    svm.fit(X_train, y_train)
    
    # Evaluate the Support Vector Machine classifier
    accuracy = svm.score(X_test, y_test)
    print("Support Vector Machine Accuracy:", accuracy)

    # Remove the . from the test_size
    test_size = str(test_size).replace('.', '_')

    # Save the model to disk
    filename = f'models/svmScaled{test_size}.sav'
    pickle.dump(svm, open(filename, 'wb'))
    
    return svm, accuracy

def create_svm_model_PCA(X_pca, y, test_size=0.1):

    """
    This function creates a Support Vector Machine classifier using the PCA features and evaluates its accuracy.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, random_state=42)
    
    # Create a Support Vector Machine classifier
    svm = SVC()
    
    # Fit the Support Vector Machine classifier to the training data
    svm.fit(X_train, y_train)
    
    # Evaluate the Support Vector Machine classifier
    accuracy = svm.score(X_test, y_test)
    print("Support Vector Machine Accuracy with PCA:", accuracy)

    # Remove the . from the test_size
    test_size = str(test_size).replace('.', '_')

    # Save the model to disk
    filename = f'models/svmPCA{test_size}.sav'
    pickle.dump(svm, open(filename, 'wb'))
    
    return svm, accuracy

#--------------------------------------------------------------------------------------------------------------#

### Naive Bayes ###

def create_naive_bayes_model(X_scaled, y, test_size=0.1):

    """
    This function creates a Naive Bayes classifier using the scaled features and evaluates its accuracy.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    # Create a Naive Bayes classifier
    naive_bayes = GaussianNB()

    # Fit the Naive Bayes classifier to the training data
    naive_bayes.fit(X_train, y_train)

    # Evaluate the Naive Bayes classifier
    accuracy = naive_bayes.score(X_test, y_test)
    print("Naive Bayes Accuracy:", accuracy)

    # Remove the . from the test_size
    test_size = str(test_size).replace('.', '_')

    # Save the model to disk
    filename = f'models/naiveBayesScaled{test_size}.sav'
    pickle.dump(naive_bayes, open(filename, 'wb'))

    return naive_bayes, accuracy

def create_naive_bayes_model_PCA(X_pca, y, test_size=0.1):

    """
    This function creates a Naive Bayes classifier using the PCA features and evaluates its accuracy.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, random_state=42)

    # Create a Naive Bayes classifier
    naive_bayes = GaussianNB()

    # Fit the Naive Bayes classifier to the training data
    naive_bayes.fit(X_train, y_train)

    # Evaluate the Naive Bayes classifier
    accuracy = naive_bayes.score(X_test, y_test)
    print("Naive Bayes Accuracy with PCA:", accuracy)

    # Remove the . from the test_size
    test_size = str(test_size).replace('.', '_')

    # Save the model to disk
    filename = f'models/naiveBayesPCA{test_size}.sav'
    pickle.dump(naive_bayes, open(filename, 'wb'))

    return naive_bayes, accuracy

#--------------------------------------------------------------------------------------------------------------#

### Random Forest Classifier ###

def create_random_forest_model(X_scaled, y, test_size=0.1):

    """
    This function creates a Random Forest classifier using the scaled features and evaluates its accuracy.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    # Create a Random Forest classifier
    random_forest = RandomForestClassifier(random_state=42)

    # Fit the Random Forest classifier to the training data
    random_forest.fit(X_train, y_train)

    # Evaluate the Random Forest classifier
    accuracy = random_forest.score(X_test, y_test)
    print("Random Forest Accuracy:", accuracy)

    # Remove the . from the test_size
    test_size = str(test_size).replace('.', '_')

    # Save the model to disk
    filename = f'models/randomForestScaled{test_size}.sav'
    pickle.dump(random_forest, open(filename, 'wb'))

    return random_forest, accuracy

def create_random_forest_model_PCA(X_pca, y, test_size=0.1):

    """
    This function creates a Random Forest classifier using the PCA features and evaluates its accuracy.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, random_state=42)

    # Create a Random Forest classifier
    random_forest = RandomForestClassifier(random_state=42)

    # Fit the Random Forest classifier to the training data
    random_forest.fit(X_train, y_train)

    # Evaluate the Random Forest classifier
    accuracy = random_forest.score(X_test, y_test)
    print("Random Forest Accuracy with PCA:", accuracy)

    # Remove the . from the test_size
    test_size = str(test_size).replace('.', '_')

    # Save the model to disk
    filename = f'models/randomForestPCA{test_size}.sav'
    pickle.dump(random_forest, open(filename, 'wb'))

    return random_forest, accuracy

#--------------------------------------------------------------------------------------------------------------#

### Logistic Regression ###

def create_logistic_regression_model(X_scaled, y, test_size=0.1):

    """
    This function creates a Logistic Regression classifier using the scaled features and evaluates its accuracy.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    # Create a Logistic Regression classifier
    logistic_regression = LogisticRegression(random_state=42)

    # Fit the Logistic Regression classifier to the training data
    logistic_regression.fit(X_train, y_train)

    # Evaluate the Logistic Regression classifier
    accuracy = logistic_regression.score(X_test, y_test)
    print("Logistic Regression Accuracy:", accuracy)

    # Remove the . from the test_size
    test_size = str(test_size).replace('.', '_')

    # Save the model to disk
    filename = f'models/logisticRegressionScaled{test_size}.sav'
    pickle.dump(logistic_regression, open(filename, 'wb'))

    return logistic_regression, accuracy

def create_logistic_regression_model_PCA(X_pca, y, test_size=0.1):

    """
    This function creates a Logistic Regression classifier using the PCA features and evaluates its accuracy.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, random_state=42)

    # Create a Logistic Regression classifier
    logistic_regression = LogisticRegression(random_state=42)

    # Fit the Logistic Regression classifier to the training data
    logistic_regression.fit(X_train, y_train)

    # Evaluate the Logistic Regression classifier
    accuracy = logistic_regression.score(X_test, y_test)
    print("Logistic Regression Accuracy with PCA:", accuracy)

    # Remove the . from the test_size
    test_size = str(test_size).replace('.', '_')

    # Save the model to disk
    filename = f'models/logisticRegressionPCA{test_size}.sav'
    pickle.dump(logistic_regression, open(filename, 'wb'))

    return logistic_regression, accuracy

#--------------------------------------------------------------------------------------------------------------#

### Neural Network ###

def create_neural_network_model(X_scaled, y, test_size=0.1):

    """
    This function creates a Neural Network classifier using the scaled features and evaluates its accuracy.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    max_iter = 1500

    # Create a Neural Network classifier
    neural_network = MLPClassifier(max_iter=max_iter, random_state=42)

    # Fit the Neural Network classifier to the training data
    neural_network.fit(X_train, y_train)

    # Evaluate the Neural Network classifier
    accuracy = neural_network.score(X_test, y_test)
    print("Neural Network Accuracy:", accuracy)

    # Remove the . from the test_size
    test_size = str(test_size).replace('.', '_')

    # Save the model to disk
    filename = f'models/neuralNetworkScaled{test_size}.sav'
    pickle.dump(neural_network, open(filename, 'wb'))

    return neural_network, accuracy

def create_neural_network_model_PCA(X_pca, y, test_size=0.1):

    """
    This function creates a Neural Network classifier using the PCA features and evaluates its accuracy.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, random_state=42)

    max_iter = 1500

    # Create a Neural Network classifier
    neural_network = MLPClassifier(max_iter=max_iter, random_state=42)

    # Fit the Neural Network classifier to the training data
    neural_network.fit(X_train, y_train)

    # Evaluate the Neural Network classifier
    accuracy = neural_network.score(X_test, y_test)
    print("Neural Network Accuracy with PCA:", accuracy)

    # Remove the . from the test_size
    test_size = str(test_size).replace('.', '_')

    # Save the model to disk
    filename = f'models/neuralNetworkPCA{test_size}.sav'
    pickle.dump(neural_network, open(filename, 'wb'))

    return neural_network, accuracy

#--------------------------------------------------------------------------------------------------------------#

### Predict Function ###
def predictStateOfGame(matchToPredict, scaler):
    """
        This function predicts the state of the game based on the data of the matches.
        
        Parameters:
            matchToPredict -> Dictionary with the information of the match to predict.
            
        Returns:
            1 if the blue team is going to win, 0 otherwise.
    """
    match_data = [
        float(matchToPredict['blueTeam']['blueTeamControlWardsPlaced']),
        float(matchToPredict['blueTeam']['blueTeamWardsPlaced']),
        float(matchToPredict['blueTeam']['blueTeamTotalKills']),
        float(matchToPredict['blueTeam']['blueTeamDragonKills']),
        float(matchToPredict['blueTeam']['blueTeamHeraldKills']),
        float(matchToPredict['blueTeam']['blueTeamTowersDestroyed']),
        float(matchToPredict['blueTeam']['blueTeamInhibitorsDestroyed']),
        float(matchToPredict['blueTeam']['blueTeamTurretPlatesDestroyed']),
        float(matchToPredict['firstBlood']),
        float(matchToPredict['blueTeam']['blueTeamMinionsKilled']),
        float(matchToPredict['blueTeam']['blueTeamJungleMinions']),
        float(matchToPredict['blueTeam']['blueTeamTotalGold']),
        float(matchToPredict['blueTeam']['blueTeamXp']),
        float(matchToPredict['blueTeam']['blueTeamTotalDamageToChamps']),
        float(matchToPredict['redTeam']['redTeamControlWardsPlaced']),
        float(matchToPredict['redTeam']['redTeamWardsPlaced']),
        float(matchToPredict['redTeam']['redTeamTotalKills']),
        float(matchToPredict['redTeam']['redTeamDragonKills']),
        float(matchToPredict['redTeam']['redTeamHeraldKills']),
        float(matchToPredict['redTeam']['redTeamTowersDestroyed']),
        float(matchToPredict['redTeam']['redTeamInhibitorsDestroyed']),
        float(matchToPredict['redTeam']['redTeamTurretPlatesDestroyed']),
        float(matchToPredict['redTeam']['redTeamMinionsKilled']),
        float(matchToPredict['redTeam']['redTeamJungleMinions']),
        float(matchToPredict['redTeam']['redTeamTotalGold']),
        float(matchToPredict['redTeam']['redTeamXp']),
        float(matchToPredict['redTeam']['redTeamTotalDamageToChamps'])
    ]

    match_data = np.array(match_data).reshape(1, -1)

    match_data = scaler.fit_transform(match_data)

    # List all files in the models directory without PCA in the name
    print("You will be asked to choose a model without PCA because of the minimum sample size (17)")
    models = [model for model in os.listdir('models') if 'PCA' not in model]

    #if there are no models, print "Please create a model first" and exit the program
    if len(models) == 0:
        print("Please create a model first")
        sys.exit()

    # Print all the models
    for i, model in enumerate(models, start=1):
        print(f"{i}. {model}")

    # Ask the user to choose a model
    model_number = int(input("Enter the number of the model you want to use: "))
    chosen_model = models[model_number - 1]

    # Load the chosen model
    filename = f'models/{chosen_model}'
    loaded_model = pickle.load(open(filename, 'rb'))

    predictions = loaded_model.predict(match_data)

    if predictions[0] == 1:
        print("The blue team is going to win")
    else:
        print("The red team is going to win")

    return predictions[0]

def predictStateOfGameFile(matchToPredictFile, scaler, pca):

    """
        This function predicts the state of the game based on the data of the matches.
    """

    match_data = pd.read_csv('predict_data/' + matchToPredictFile, skiprows=1, names= [
        'blueTeamControlWardsPlaced',
        'blueTeamWardsPlaced',
        'blueTeamTotalKills',
        'blueTeamDragonKills',
        'blueTeamHeraldKills',
        'blueTeamTowersDestroyed',
        'blueTeamInhibitorsDestroyed',
        'blueTeamTurretPlatesDestroyed',
        'firstBlood',
        'blueTeamMinionsKilled',
        'blueTeamJungleMinions',
        'blueTeamTotalGold',
        'blueTeamXp',
        'blueTeamTotalDamageToChamps',
        'redTeamControlWardsPlaced',
        'redTeamWardsPlaced',
        'redTeamTotalKills',
        'redTeamDragonKills',
        'redTeamHeraldKills',
        'redTeamTowersDestroyed',
        'redTeamInhibitorsDestroyed',
        'redTeamTurretPlatesDestroyed',
        'redTeamMinionsKilled',
        'redTeamJungleMinions',
        'redTeamTotalGold',
        'redTeamXp',
        'redTeamTotalDamageToChamps'
    ])

    match_data = scaler.fit_transform(match_data)

    # List all files in the models directory
    models = os.listdir('models')

    #if there are no models, print "Please create a model first" and exit the program
    if len(models) == 0:
        print("Please create a model first")
        sys.exit()

    isFile = True

    with open('predict_data/' + matchToPredictFile, 'r') as file:
        lines = file.readlines()
        if len(lines) < 18:
            isFile = False

    # Explain that if isFile is false, the user will be asked to choose a model without PCA because of the minimum sample size (17) - also do not list models with PCA in the name
    if not isFile:
        print("You will be asked to choose a model without PCA because of the minimum sample size (17)")
        models = [model for model in models if 'PCA' not in model]

    # Print all the models
    for i, model in enumerate(models, start=1):
        print(f"{i}. {model}")

    # Ask the user to choose a model
    model_number = int(input("Enter the number of the model you want to use: "))
    chosen_model = models[model_number - 1]

    # Load the chosen model
    filename = f'models/{chosen_model}'
    loaded_model = pickle.load(open(filename, 'rb'))

    # if model has PCA in name, perform PCA on the match data before predicting, else predict directly
    if 'PCA' in chosen_model:
        pca = PCA(n_components=17)
        match_data = pca.fit_transform(match_data)

    predictions = loaded_model.predict(match_data)

    # print the prediction for each match, with the format "Match X: The blue team is going to win" or "Match X: The red team is going to win", where X starts at 1
    for i, prediction in enumerate(predictions, start=1):
        if prediction == 1:
            print(f"Match {i}: The blue team is going to win")
        else:
            print(f"Match {i}: The red team is going to win")

    noCsv = matchToPredictFile.split('.')[0]

    # output the prediction to a file called matchToPredictFile_predictions.txt, where each line contains the prediction for each match
    with open('predict_data/' + noCsv + '_output.txt', 'w') as file:
        for prediction in predictions:
            if prediction == 1:
                file.write("1\n")
            else:
                file.write("0\n")

    return 0

#--------------------------------------------------------------------------------------------------------------#    

### Main Function ###
if __name__ == "__main__":
    menu()
    
#--------------------------------------------------------------------------------------------------------------#    