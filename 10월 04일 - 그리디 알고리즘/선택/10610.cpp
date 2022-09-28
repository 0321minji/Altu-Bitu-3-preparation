#include<iostream>
#include<algorithm>
#include<string>
using namespace std;


/*
[���� ū ��]
-�Է� ���� �� ���� ���� ū ������� ����

[30�� ��� ����]
-3�� ��� : �� �ڸ��� ���� ���� 3�� ���
-10�� ��� : ���� �ڸ��� 0
*/

//��� �ڸ��� ���� 3�� ������� Ȯ���ϴ� �Լ�
bool checkThree(string num) {
    int sum = 0;
    for (int i = 0; i < num.size(); i++) {
        sum += num[i];
    }

    //���� �ڸ����� ���� 3�� ����̸� num�� 3�� ���
    if (sum % 3 == 0) {
        return true;
    }
    return false;
}

//n���� ���� �� �ִ� 30�� ��� �� ���� ū ���� ���ϴ� �Լ�
string findNum(string n) {
    sort(n.begin(), n.end(), greater<>()); //�� �ڸ��� �������� ����

    //30�� ����̸� ���� ����
    if (n[n.size() - 1] == '0' && checkThree(n)) {
        return n;
    }
    return "-1";

}

int main() {
    string n;
    cin >> n;

    string ans = findNum(n);

    //N�� �ִ� 10^5���� ���ڷ� �����Ǿ� �����Ƿ� stringŸ������ �ϳ��� ���
    for (int i = 0; i < ans.size(); i++) {
        cout << ans[i];
    }
    cout << '\n';
    return 0;

}