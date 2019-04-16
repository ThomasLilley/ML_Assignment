import Task_1
import Task_2


def menu():
    flag = True
    while flag:
        print('\nMain Menu')
        print('---------')
        print('1) Task 1')
        print('2) Task 2')
        print('3) Exit')
        print('---------')
        a = input('Please select an option: ')

        if a == '1':
            Task_1.task1()
        elif a == '2':
            Task_2.task2()
        elif a == '3':
            exit(0)
        else:
            print('Invalid option selected')


menu()
