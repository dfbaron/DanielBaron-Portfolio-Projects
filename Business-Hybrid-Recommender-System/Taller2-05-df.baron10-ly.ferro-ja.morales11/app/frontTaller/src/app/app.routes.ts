import { RouterModule, Routes } from '@angular/router';
import { ListaRecomendacionesComponent } from './lista-recomendaciones/lista-recomendaciones.component';
import { LoginComponent } from './login/login.component';
import { UsuarioComponent } from './usuario/usuario.component';


const ROUTES: Routes = [
    { path: "usuario", component: UsuarioComponent},
    { path: "login", component: LoginComponent},
    { path: "recomendaciones", component: ListaRecomendacionesComponent},
    { path: "**", pathMatch: "full", redirectTo: "login"},
    
  ];

  export const APP_ROUTING = RouterModule.forRoot(ROUTES);