from firebase import firebase

firebase = firebase.FirebaseApplication('https://door-unlock-29cbb-default-rtdb.firebaseio.com/',None)
result = firebase.get('/value/','')
data={'value':1}
firebase.put('','value',1)
print(result)