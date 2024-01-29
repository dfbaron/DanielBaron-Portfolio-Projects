import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { UsuarioService } from '../services/usuario.service';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {

  constructor(public usuarioService: UsuarioService, private router: Router) { }

  ngOnInit(): void {
  }

  login(){
    console.log(this.usuarioService.idLogged);

    this.usuarioService.get_usuario(this.usuarioService.idLogged).subscribe((data: any) => {
      if (data.length == 0){
        console.log("No existe este usuario");
      } else {
        this.usuarioService.setLogStatus(true);
        this.router.navigate(['/usuario'])
      }
    });
}

}
