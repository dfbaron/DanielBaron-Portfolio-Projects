import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

const API_URL = "http://172.24.41.184:8081/"

@Injectable({
  providedIn: 'root'
})
export class UsuarioService {

  loggedIn : boolean = false;

  idLogged: string = "";

  constructor( private http: HttpClient) { 
    console.log("Usuario API lista")
  }

  httpOptions = {
    headers: new HttpHeaders({ "Content-Type": "application/json" })
  };

  httpOptions2 = {
    headers: new HttpHeaders({
        'Accept': 'text/html',
        'Content-Type': 'application/json',
        'responseType': 'text'
    }),
  };




  get_usuario(id:string){
    return this.http.get(API_URL + 'get_usuario/' + id);
  }

  get_recomendaciones_by_id(id:string){
    return this.http.get(API_URL + 'get_recomendaciones/' + id, this.httpOptions);
  }

  get_mapa(recomendaciones: any[]) : Observable<any>{
    return this.http.post(API_URL + 'get_mapa', recomendaciones, {'responseType': 'text'});
  }

  getLogStatus (){
    return this.loggedIn;
  }

  setLogStatus (status: boolean){
    
    this.loggedIn = status;
  }
}
