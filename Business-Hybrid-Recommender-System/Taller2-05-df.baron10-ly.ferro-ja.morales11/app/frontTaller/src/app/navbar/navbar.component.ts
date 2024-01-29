import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { UsuarioService } from '../services/usuario.service';

@Component({
  selector: 'app-navbar',
  templateUrl: './navbar.component.html',
  styleUrls: ['./navbar.component.css']
})
export class NavbarComponent implements OnInit {

  constructor(public usuarioService : UsuarioService, private router: Router) { }

  ngOnInit(): void {
  }

  logout(){
    this.usuarioService.setLogStatus(false);
    this.usuarioService.idLogged = "";
    this.router.navigate(['/login'])
  }

}
