#include<iostream>
#include<vector>
#include<cmath>
#include<set>
using namespace std;

void isPrime(int n, vector<bool> &is_prime) {//�����佺�׳׽��� ü 
	//0�� 1�� �Ҽ��� �ƴϹǷ� ���� ����
	is_prime[0] = is_prime[1] = false;
	//2~������ n���� �˻�
	for (int i = 2; i <= sqrt(n); i++) {
		if (is_prime[i]) {//i�� �Ҽ����
			for (int j = i * i; j <= n; j += i) {
				is_prime[j] = false;//i�� ����� ����
			}
		}
	}
}


int calSumNum(int n) {//n�� �� �ڸ����� ������ ���� ��  sum�� ��ȯ�ϴ� �Լ�
	int sum = 0;
	while (n) {
		sum += pow(n % 10, 2);
		n /= 10;
	}
	return sum;
}

void isAnswer(int n, vector<bool>& is_prime, vector<int>& answer) {
	//���� �Ǳ� ���ؼ��� �Ҽ��̸鼭 ��ټ����� �ؿ�. �Ҽ����� �Ǵ��� ����� is_prime�� ����Ǿ���ϴ�.
	//���� n�� �� �ڸ����� ���ϴ� ����� �ݺ��ϸ鼭 ����� 1�� ������ ��ټ��̰� �� ���� ���� ����� �ٽ� ���´ٸ� ��ټ��� �ƴϿ���
	
	set<int> s;//���� ���� ����� �����ϰ� Ž������ �� �� �ִ� set�� ����ؿ�

	for (int i = 2; i < n; i ++ ) {
		if (is_prime[i]) { //i�� �Ҽ����
			int tmp = i; //i�� �ݺ��� ���ǿ� ���ǹǷ� ���� ��ȭ�� ���� �ϱ� ���� ���ο� ������ �����ؿ�
			while (1) {

				tmp = calSumNum(tmp); //calSumNum�Լ��� ���� �� �ڸ��� ������ ���� tmp�� �����ؿ�

				if (tmp == 1) {//�ڸ����� ���� 1�̸�
					answer.push_back(i);//�信 �����ϰ�
					break;//�ݺ����� ����������
				}
				
				if (s.find(tmp) != s.end()) { //�ڸ��� ������ ���� �� ���� ����� ���ٸ�
					break;//�ݺ����� ����������
				}
				s.insert(tmp);//�ڸ����� ���� 1�� �ƴϰ� �� ����� ��ġ�� ������ s�� �����ϰ� �ݺ����� ó������ ���ư���
			}
		}
		s.clear();//���� ������ �Ǵ��ϱ� ���� �� s�� ������ؿ�
	}
}


int main() {
	int n;

	cin >> n;

	vector<bool> is_prime(n + 1, true); // �Ҽ����� �Ǵ��� bool�� ������ �迭
	vector<int> answer; //�Ҽ� ��ټ� ������ �迭

	isPrime(n, is_prime);//�Ҽ����� �Ǵ�
	isAnswer(n, is_prime, answer);//�Ҽ� ��ټ����� �Ǵ�

	for (int i = 0; i < answer.size(); i++) {
		cout << answer[i] << '\n';
	}

	return 0;
}